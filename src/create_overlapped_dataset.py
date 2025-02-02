import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Dict
from uuid import uuid4

import librosa
import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torchaudio.functional as F
from datasets import load_dataset
from tqdm import tqdm

from utils.lufs_utils import calculate_lufs, get_lufs_norm_audio
from utils.opuslib_module import (
    OpusBytesEncoderDecoder,
    postprocess_waveform,
    preprocess_waveform,
)


def normalize_to_audio1_power(audio1, audio2):
    """Scales both audios so that they have the power of the weaker one."""
    power1 = np.mean(audio1**2)
    power2 = np.mean(audio2**2)

    audio2 = audio2 * np.sqrt(power1 / power2)  # Scale audio2 to weaker power
    #print(f"after norm: a1 power:{np.mean(audio1**2)}, a2power:{np.mean(audio2**2)}")

    return audio2


def normalize_to_weaker_power(audio1, audio2):
    """Scales both audios so that they have the power of the weaker one."""
    power1 = np.mean(audio1**2)
    power2 = np.mean(audio2**2)

    weaker_power = min(power1, power2)  # Find the weaker power

    if power1 > 0:
        audio1 = audio1 * np.sqrt(weaker_power / power1)  # Scale audio1 to weaker power

    if power2 > 0:
        audio2 = audio2 * np.sqrt(weaker_power / power2)  # Scale audio2 to weaker power
    # print(f"after norm: a1 power:{np.mean(audio1**2)}, a2power:{np.mean(audio2**2)}")
    return audio1, audio2


def normalize_mean(audio):
    audio = audio - np.mean(audio)
    return audio


def will_clipping_occur(audio1, audio2, scaling_factor):
    """
    Check if mixing two waveforms with a given scaling factor will cause clipping.

    Parameters:
    - audio1 (np.ndarray): First waveform.
    - audio2 (np.ndarray): Second waveform.
    - scaling_factor (float): Scaling factor applied to the second waveform.

    Returns:
    - bool: True if clipping will occur, False otherwise.
    - float: The predicted maximum absolute value after mixing.
    """
    # Calculate the predicted mixed waveform without actually creating it
    predicted_max = np.max(np.abs(audio1)) + (scaling_factor * np.max(np.abs(audio2)))

    # Check if the predicted max exceeds the valid range
    clipping_occurred = predicted_max > 1.0
    return clipping_occurred, predicted_max


def add_music_to_mixed_file(music, wav_file, music_scale: int):
    if len(music) > len(wav_file):
        music = music[: len(wav_file)]  # Truncate
    else:
        padding = np.zeros(len(wav_file) - len(music))
        music = np.concatenate((music, padding))  # Pad with zeros

    # Sum the two signals
    combined_audio = wav_file + music * music_scale
    # Normalize the combined audio to avoid clipping
    combined_audio = normalize_mean(combined_audio)
    return combined_audio


def add_noise_to_match_snr(audio, snr_db: int):
    """
    Adds noise to an audio signal to achieve the desired SNR.

    Parameters:
        audio (numpy.ndarray): Audio signal.
        snr_db (float): Desired SNR in dB.

    Returns:
        numpy.ndarray: Noisy audio signal.
    """
    # Calculate the power of the audio signal
    signal_power = np.mean(audio**2)

    # Calculate the desired noise power to achieve the given SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_amplitude = np.sqrt(noise_power)
    # Generate white Gaussian noise with the calculated power
    noise = noise_amplitude * np.random.normal(0, 1, len(audio))

    # Add noise to the original audio
    noisy_audio = audio + noise
    # noisy_audio = normalize_audio(noisy_audio)
    # return noisy audio and also linear snr for metadata saving
    return noisy_audio, snr_linear, noise_amplitude


def load_random_wav(directory: str, target_sample_rate: int):
    """Choose a random RIR from a directory."""
    wav_files = [f for f in os.listdir(directory) if f.endswith(".wav")]
    random_wav_str = random.choice(wav_files)
    wav_path = os.path.join(directory, random_wav_str)
    waveform, wav_sample_rate = librosa.load(wav_path, sr=None)
    if wav_sample_rate != target_sample_rate:
        waveform = librosa.resample(
            waveform, orig_sr=wav_sample_rate, target_sr=target_sample_rate
        )
    return waveform, random_wav_str


def apply_rir_to_audio(audio, rir):
    """Convolve the audio with the RIR."""
    # convolved_audio = scipy.signal.convolve(audio, rir, mode="full")
    convolved_audio = scipy.signal.fftconvolve(audio, rir, mode="full")
    return convolved_audio


def calc_scale_factor(audio1, audio2, sns_db_scale: int):
    """Calc power of each singal."""
    # Calculate the power of the audio signal
    signal_power_audio1 = np.mean(audio1**2)
    signal_power_audio2 = np.mean(audio2**2)

    # Calculate the desired noise power to achieve the given SNR
    snr_linear = 10 ** (sns_db_scale / 10)
    # mixture = audio1 + a*audio2
    # snr_linear = power(y1)/(a^2*power(y2))
    a = np.sqrt(signal_power_audio1 / (signal_power_audio2 * snr_linear))
    return a, signal_power_audio1, signal_power_audio2, snr_linear


def peak_normalize(audio, target_peak=0.98):
    """Normalize audio so that the maximum absolute amplitude equals target_peak.

    Returns:
        - normalized_audio (np.ndarray): Peak normalized audio
        - scale_factor (float): Multiplication factor used for normalization
    """
    max_amplitude = np.max(np.abs(audio))

    if max_amplitude > 0:  # Avoid division by zero
        scale_factor = target_peak / max_amplitude
        audio = audio * scale_factor
    else:
        scale_factor = 1.0  # No scaling if max amplitude is 0 (silence)

    return audio, scale_factor


def reduce_high_peaks(audio, threshold=0.9):
    """Reduce high peaks in the audio signal."""
    peak_amplitude = np.max(np.abs(audio))
    if peak_amplitude > threshold:
        audio = audio / peak_amplitude * threshold
    return audio


def arrange_cutoff_freq_audios_by_power(audio1, audio2, freq1, freq2):
    """Arrange audios by signal power: stronger first, weaker second."""
    # Calculate the power of each audio signal
    signal_power_audio1 = np.mean(audio1**2)
    signal_power_audio2 = np.mean(audio2**2)

    # Return the audios sorted by descending power
    if signal_power_audio1 >= signal_power_audio2:
        return freq1, freq2
    else:
        return freq2, freq1


def mix_audio(
    file1_path: str,
    file2_path: str,
    transcription1: str,
    transcription2: str,
    target_sample_rate: int,
    max_noise_desired_snr: int,
    min_noise_desired_snr: int,
    max_music_ssr: int,
    min_music_ssr: int,
    max_conversation_desired_ssr: int,
    min_conversation_desired_ssr: int,
    output_path: str,
    metadata,
    rir_directory: str,
    music_directory: str,
    compression: Dict[str, str | bool],
    low_pass_filter_config: Dict[str, bool | int],
    split: str,
    normalize_lufs: bool = False,
):
    # sample optional bitr_rate_from _range
    # will be used for this mixture creation
    sampled_bit_rate = random.choice(
        range(compression["min_bitrate"], compression["max_bitrate"] + 1000, 1000)
    )
    compression_encoder_decoder = OpusBytesEncoderDecoder(
        bitrate=sampled_bit_rate,
        bit_depth=compression["compression_config"]["bit_depth"],
    )
    compression_encoder_decoder.reset_state(
        sample_rate=config["target_sample_rate"],
        config=compression["compression_config"],
    )

    # Generate unique ID and save mixed audio
    unique_id = str(uuid4())
    # print(f"starting:{unique_id}", flush=True)
    subdirectory_path = os.path.join(output_path, unique_id)
    os.makedirs(subdirectory_path, exist_ok=True)
    # create directories for training
    source1_path = os.path.join(output_path, split, "source1")
    os.makedirs(source1_path, exist_ok=True)
    source2_path = os.path.join(output_path, split, "source2")
    os.makedirs(source2_path, exist_ok=True)
    source1_reverb_path = os.path.join(output_path, split, "source1_reverb")
    os.makedirs(source1_reverb_path, exist_ok=True)
    source2_reverb_path = os.path.join(output_path, split, "source2_reverb")
    os.makedirs(source2_reverb_path, exist_ok=True)
    mixture_path = os.path.join(output_path, split, "mixture")
    os.makedirs(mixture_path, exist_ok=True)
    compressed_mixture_path = os.path.join(output_path, split, "compressed_mixture")
    os.makedirs(compressed_mixture_path, exist_ok=True)
    # Load audio files
    y1, sr1 = librosa.load(file1_path, sr=None)
    y2, sr2 = librosa.load(file2_path, sr=None)

    if sr1 != target_sample_rate:
        y1 = librosa.resample(y1, orig_sr=sr1, target_sr=target_sample_rate)

    if sr2 != target_sample_rate:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=target_sample_rate)

    # Calculate original lengths
    length1 = len(y1) / target_sample_rate
    length2 = len(y2) / target_sample_rate

    # Pad the shorter file
    if len(y1) > len(y2):
        pad_length = (len(y1) - len(y2)) / target_sample_rate
        y2 = np.pad(y2, (0, len(y1) - len(y2)), mode="constant")
        padding1, padding2 = 0, pad_length
    else:
        pad_length = (len(y2) - len(y1)) / target_sample_rate
        y1 = np.pad(y1, (0, len(y2) - len(y1)), mode="constant")
        padding1, padding2 = pad_length, 0

    y1 = normalize_mean(y1)
    y2 = normalize_mean(y2)
    y1, _ = peak_normalize(audio=y1, target_peak=0.5)
    y2, _ = peak_normalize(audio=y2, target_peak=0.5)
    y1, y2 = normalize_to_weaker_power(audio1=y1, audio2=y2)
    # validationg equal lengths of all sources - clean and reverbant

    y1_lufs = calculate_lufs(y1, sr=target_sample_rate)
    y2_lufs = calculate_lufs(y2, sr=target_sample_rate)
    # apply Room impulse on audio + scaling and save file
    rir, _ = load_random_wav(
        directory=rir_directory, target_sample_rate=target_sample_rate
    )

    ff_audio1 = apply_rir_to_audio(y1, rir)
    ff_audio2 = apply_rir_to_audio(y2, rir)
    ff_audio1 = normalize_mean(ff_audio1)
    ff_audio2 = normalize_mean(ff_audio2)
    ff_audio1, _ = peak_normalize(audio=ff_audio1, target_peak=0.5)
    ff_audio2, _ = peak_normalize(audio=ff_audio2, target_peak=0.5)
    ff_audio1, ff_audio2 = normalize_to_weaker_power(audio1=ff_audio1, audio2=ff_audio2)

    # Ensure original sources are padded to match the length of the reverberant versions
    max_len = max(len(ff_audio1), len(ff_audio2))
    y1 = np.pad(y1, (0, max_len - len(y1)), mode="constant")
    y2 = np.pad(y2, (0, max_len - len(y2)), mode="constant")
    ff_audio1 = np.pad(ff_audio1, (0, max_len - len(ff_audio1)), mode="constant")
    ff_audio2 = np.pad(ff_audio2, (0, max_len - len(ff_audio2)), mode="constant")

    # Save original files
    original_file1 = os.path.join(subdirectory_path, os.path.basename(file1_path))
    original_file2 = os.path.join(subdirectory_path, os.path.basename(file2_path))
    sf.write(original_file1, y1, target_sample_rate)
    sf.write(original_file2, y2, target_sample_rate)
    # print(f"ff_audio1_before_noise_power:{np.mean(ff_audio1**2)}")
    # print(f"ff_audio2_before_noise_power:{np.mean(ff_audio2**2)}")
    # save far field audios
    ff_original_file1 = os.path.join(
        subdirectory_path, "ff_" + os.path.basename(file1_path)
    )
    ff_original_file2 = os.path.join(
        subdirectory_path, "ff_" + os.path.basename(file2_path)
    )
    sf.write(ff_original_file1, ff_audio1, target_sample_rate)
    sf.write(ff_original_file2, ff_audio2, target_sample_rate)
    if normalize_lufs:
        ff_audio1, gain1 = get_lufs_norm_audio(
            audio=ff_audio1.squeeze(), sr=target_sample_rate, lufs=-33
        )
        ff_audio2, gain2 = get_lufs_norm_audio(
            audio=ff_audio2.squeeze(), sr=target_sample_rate, lufs=-33
        )
    # Sample scale factor between audios
    scale_factor_DB = np.random.uniform(
        min_conversation_desired_ssr, max_conversation_desired_ssr
    )

    (
        linear_mult_factor,
        ff_signal_power_audio1,
        ff_signal_power_audio2,
        linear_scale_factor,
    ) = calc_scale_factor(
        audio1=ff_audio1, audio2=ff_audio2, sns_db_scale=scale_factor_DB
    )
    clipping, predicted_max = will_clipping_occur(
        ff_audio1, ff_audio2, linear_mult_factor
    )
    if clipping:
        print(
            f"speaker scaling - Clipping will occur! Predicted max value: {predicted_max:.2f},linear_mult_factor:{linear_mult_factor} ,uuid:{unique_id}"
        )
    # else:
    #     print(
    #         f"speaker scaling - No clipping expected. Predicted max value: {predicted_max:.2f}"
    #     )

    freq1 = None
    freq2 = None

    # apply low pass filter and save filtered far-field audios
    if low_pass_filter_config["apply_low_pass_filter"]:
        freq_1, freq_2 = arrange_cutoff_freq_audios_by_power(
            ff_audio1,
            linear_mult_factor * ff_audio2,
            low_pass_filter_config["cutoff_freq"],
            low_pass_filter_config["cutoff_freq"]
            + random.choice(range(500, 1100, 100)),
        )
        ff_audio1 = F.lowpass_biquad(
            waveform=torch.tensor(ff_audio1, dtype=torch.float32),
            sample_rate=target_sample_rate,
            cutoff_freq=torch.tensor(float(freq_1), dtype=torch.float32),
        )
        ff_audio1 = ff_audio1.numpy()

        ff_audio2 = F.lowpass_biquad(
            waveform=torch.tensor(ff_audio2, dtype=torch.float32),
            sample_rate=target_sample_rate,
            cutoff_freq=torch.tensor(float(freq_2), dtype=torch.float32),
        )
        ff_audio2 = ff_audio2.numpy()

    # Mix the audio

    ff_mixed_audio = ff_audio1 + linear_mult_factor * ff_audio2
    ff_mixed_audio = normalize_mean(ff_mixed_audio)
    ff_mixed_audio, norm_factor_ff_mixture = peak_normalize(
        ff_mixed_audio, target_peak=0.5
    )
    mixture_before_music_lufs = calculate_lufs(ff_mixed_audio, sr=target_sample_rate)
    # print(f"mixture_before_noise_power:{np.mean(ff_mixed_audio**2)}")

    ff_scaled_file1 = os.path.join(
        subdirectory_path, "ff_scaled_filtered_" + os.path.basename(file1_path)
    )
    ff_scaled_file2 = os.path.join(
        subdirectory_path, "ff_scaled_filtered_" + os.path.basename(file2_path)
    )
    mixed_audio_ff_file = os.path.join(subdirectory_path, f"ff_{unique_id}.wav")

    sf.write(ff_scaled_file1, norm_factor_ff_mixture * ff_audio1, target_sample_rate)
    sf.write(
        ff_scaled_file2,
        linear_mult_factor * norm_factor_ff_mixture * ff_audio2,
        target_sample_rate,
    )
    sf.write(mixed_audio_ff_file, ff_mixed_audio, target_sample_rate)

    # Save decompressed far-field audio
    if compression["apply_compression"]:
        mixed_audio_ff_compressed_file = os.path.join(
            subdirectory_path, f"ff_{unique_id}_opus.wav"
        )
        pcm_frames = preprocess_waveform(
            ff_mixed_audio,
            target_sample_rate,
            compression["compression_config"]["sample_rate"],
            compression["compression_config"],
        )
        if not pcm_frames or len(pcm_frames) == 0:
            raise ValueError("PCM frames are empty or invalid.")

        decoded_pcm = []
        frame_size = (
            compression["compression_config"]["sample_rate"] // 1000
        ) * compression["compression_config"]["frame_duration_ms"]

        for chunk in pcm_frames:
            # Encode and decode each frame
            encoded_data = compression_encoder_decoder.encode(
                input_data=chunk, frame_size=frame_size
            )
            decoded_frame = compression_encoder_decoder.decode(
                input_data=encoded_data, frame_size=frame_size
            )
            decoded_pcm.append(decoded_frame)

        # Combine and postprocess the decoded audio
        decoded_pcm = b"".join(decoded_pcm)
        decoded_mixed_audio_ff_file = postprocess_waveform(
            decoded_pcm, compression["compression_config"]["opus_channels_number"]
        )
        # print(f"Decoded PCM length: {len(decoded_pcm)}")

        # Postprocess
        decoded_mixed_audio_ff_file = postprocess_waveform(
            decoded_pcm, compression["compression_config"]["opus_channels_number"]
        )
        if len(decoded_mixed_audio_ff_file) > 0:
            sf.write(
                mixed_audio_ff_compressed_file,
                decoded_mixed_audio_ff_file,
                target_sample_rate,
            )
            # print(f"Compressed file saved: {mixed_audio_ff_compressed_file}")
        else:
            print("Decoded mixed audio is empty; skipping file saving.")

    # Make audios noisy
    snr_db = np.random.uniform(min_noise_desired_snr, max_noise_desired_snr)
    noisy_ff_mixed_audio, snr_linear, noise_amplitude = add_noise_to_match_snr(
        ff_mixed_audio, snr_db
    )
    # print(f"noisy_mixture_power:{np.mean(noisy_ff_mixed_audio**2)}")
    mixture_before_music_lufs = calculate_lufs(
        noisy_ff_mixed_audio, sr=target_sample_rate
    )

    # Save noisy far-field audio
    noisy_ff_mixed_audio_file_path = os.path.join(
        subdirectory_path, f"{unique_id}_noisy.wav"
    )
    sf.write(noisy_ff_mixed_audio_file_path, noisy_ff_mixed_audio, target_sample_rate)

    # add random noise from music directory and save file
    additional_music_wav, addition_music_str = load_random_wav(
        directory=music_directory, target_sample_rate=target_sample_rate
    )
    additional_music_wav = normalize_mean(additional_music_wav)
    additional_music_wav, _ = peak_normalize(additional_music_wav, target_peak=0.5)
    ff_additional_music_wav = apply_rir_to_audio(additional_music_wav, rir)
    ff_additional_music_wav = normalize_mean(ff_additional_music_wav)
    ff_additional_music_wav, _ = peak_normalize(
        ff_additional_music_wav, target_peak=0.5
    )
    ff_additional_music_wav = normalize_to_audio1_power(
        audio1=noisy_ff_mixed_audio, audio2=ff_additional_music_wav
    )
    # print(f"music_power:{np.mean(ff_additional_music_wav**2)}")
    # save additional music
    additional_music_path = os.path.join(subdirectory_path, addition_music_str)
    additional_ff_music_path = os.path.join(
        subdirectory_path, "ff_" + addition_music_str
    )

    sf.write(additional_music_path, additional_music_wav, target_sample_rate)
    sf.write(additional_ff_music_path, ff_additional_music_wav, target_sample_rate)

    # apply filter on music
    if low_pass_filter_config["apply_low_pass_filter"]:
        ff_additional_music_wav = F.lowpass_biquad(
            waveform=torch.tensor(ff_additional_music_wav, dtype=torch.float32),
            sample_rate=target_sample_rate,
            cutoff_freq=torch.tensor(float(freq_2), dtype=torch.float32),
        )
        ff_additional_music_wav = ff_additional_music_wav.numpy()

    ff_music_lufs = calculate_lufs(ff_additional_music_wav, sr=target_sample_rate)
    if normalize_lufs:
        ff_additional_music_wav, music_gain2 = get_lufs_norm_audio(
            audio=ff_additional_music_wav.squeeze(), sr=target_sample_rate, lufs=-40
        )

    # Sample scale factor between audios
    mix2music_snr_DB = np.random.uniform(min_music_ssr, max_music_ssr)
    (
        music_linear_mult_factor,
        mixture_power_before_music,
        music_signal_power,
        mix2music_snr_linear,
    ) = calc_scale_factor(
        audio1=noisy_ff_mixed_audio,
        audio2=ff_additional_music_wav,
        sns_db_scale=mix2music_snr_DB,
    )

    clipping, predicted_max = will_clipping_occur(
        noisy_ff_mixed_audio, ff_additional_music_wav, music_linear_mult_factor
    )
    if clipping:
        print(
            f"music - Clipping will occur adding music! Predicted max value:{predicted_max:.2f},music_linear_mult_factor:{music_linear_mult_factor} uuid:{unique_id}"
        )
    # else:
    # print(
    #    f"speaker scaling - No clipping expected. Predicted max value: {predicted_max:.2f}"
    # )
    # save scaled and filtered additional music
    additional_ff_scaled_filtered_music_path = os.path.join(
        subdirectory_path, "ff_scaled_filtered_" + addition_music_str
    )
    sf.write(
        additional_ff_scaled_filtered_music_path,
        music_linear_mult_factor * ff_additional_music_wav,
        target_sample_rate,
    )
    noisy_ff_with_music_mixed_audio = add_music_to_mixed_file(
        music=ff_additional_music_wav,
        wav_file=noisy_ff_mixed_audio,
        music_scale=music_linear_mult_factor,
    )
    # print(
    #    f"noisy_ff_with_music_mixed_audio_power:{np.mean(noisy_ff_with_music_mixed_audio**2)}"
    # )
    if np.max(noisy_ff_with_music_mixed_audio) > 0.9:
        print(f"mixture_before_music_lufs:{mixture_before_music_lufs}")
        print(f"mixture_power_before_music:{mixture_power_before_music}")
        print(f"music_signal_power:{music_signal_power}")
        print(f"mix2music_snr_linear:{mix2music_snr_linear}")
        print(f"music_linear_mult_factor:{music_linear_mult_factor}")
        print(f"mix2music_snr_DB:{mix2music_snr_DB}")
        print(f"ff_music_lufs:{ff_music_lufs}")
    # save far-field mix with noise and overlapped music
    mixed_audio_ff_file_with_music_path = os.path.join(
        subdirectory_path, f"ff_{unique_id}_noisy_with_music.wav"
    )
    sf.write(
        mixed_audio_ff_file_with_music_path,
        noisy_ff_with_music_mixed_audio,
        target_sample_rate,
    )
    # Save compressed mix with noise and overlapped music
    if compression["apply_compression"]:
        noisy_ff_mixed_audio_with_music_compressed_file_path = os.path.join(
            subdirectory_path, f"ff_{unique_id}_noisy_with_music_opus.wav"
        )
        pcm_frames = preprocess_waveform(
            noisy_ff_with_music_mixed_audio,
            target_sample_rate,
            compression["compression_config"]["sample_rate"],
            compression["compression_config"],
        )
        if not pcm_frames or len(pcm_frames) == 0:
            raise ValueError("PCM frames are empty or invalid.")

        decoded_pcm = []
        frame_size = (
            compression["compression_config"]["sample_rate"] // 1000
        ) * compression["compression_config"]["frame_duration_ms"]

        for chunk in pcm_frames:
            # Encode and decode each frame
            encoded_data = compression_encoder_decoder.encode(
                input_data=chunk, frame_size=frame_size
            )
            decoded_frame = compression_encoder_decoder.decode(
                input_data=encoded_data, frame_size=frame_size
            )
            decoded_pcm.append(decoded_frame)

        # Combine and postprocess the decoded audio
        decoded_pcm = b"".join(decoded_pcm)
        decoded_mixed_audio_ff_file = postprocess_waveform(
            decoded_pcm, compression["compression_config"]["opus_channels_number"]
        )
        # print(f"Decoded PCM length: {len(decoded_pcm)}")

        # Postprocess
        decoded_noisy_mixed_audio_ff_with_music_file = postprocess_waveform(
            decoded_pcm, compression["compression_config"]["opus_channels_number"]
        )
        sf.write(
            noisy_ff_mixed_audio_with_music_compressed_file_path,
            decoded_noisy_mixed_audio_ff_with_music_file,
            target_sample_rate,
        )

    # save files to training dir

    source_1_path = os.path.join(output_path, split, "source1", f"{unique_id}.wav")
    source_2_path = os.path.join(output_path, split, "source2", f"{unique_id}.wav")
    source_1_reverb_path = os.path.join(
        output_path, split, "source1_reverb", f"{unique_id}.wav"
    )
    source_2_reverb_path = os.path.join(
        output_path, split, "source2_reverb", f"{unique_id}.wav"
    )
    mixture_path = os.path.join(output_path, split, "mixture", f"{unique_id}.wav")
    compressed_mixture_path = os.path.join(
        output_path, split, "compressed_mixture", f"comp_{unique_id}.wav"
    )

    sf.write(source_1_path, y1, target_sample_rate)
    sf.write(source_2_path, y2, target_sample_rate)
    sf.write(
        source_1_reverb_path, norm_factor_ff_mixture * ff_audio1, target_sample_rate
    )
    sf.write(
        source_2_reverb_path,
        norm_factor_ff_mixture * linear_mult_factor * ff_audio2,
        target_sample_rate,
    )
    sf.write(mixture_path, noisy_ff_with_music_mixed_audio, target_sample_rate)
    sf.write(
        compressed_mixture_path,
        decoded_noisy_mixed_audio_ff_with_music_file,
        target_sample_rate,
    )

    # print(f"finished:{unique_id}", flush=True)
    # print("------------------------------------")

    # Save metadata
    metadata_entry = {
        "id": unique_id,
        "length_seconds": len(ff_mixed_audio) / target_sample_rate,
        "added_noise_snr_db": round(snr_db, 3),
        "added_noise_snr_linear": round(float(snr_linear), 3),
        "noise_amplitude": round(float(noise_amplitude), 3),
        "mix2music_snr_DB": round(mix2music_snr_DB, 3),
        "mix2music_snr_linear": round(float(mix2music_snr_linear), 3),
        "mixture_power_before_music": round(float(mixture_power_before_music), 7),
        "mixture_before_music_lufs": mixture_before_music_lufs,
        "norm_factor_ff_mixture": round(float(norm_factor_ff_mixture), 3),
        "music_signal_power": round(float(music_signal_power), 7),
        "music_linear_mult_factor": round(float(music_linear_mult_factor), 3),
        "additional_music": addition_music_str,
        "music_lufs": ff_music_lufs,
        "compression_bit_rate": sampled_bit_rate,
        "original_files": [
            {
                "file": os.path.basename(file1_path),
                "original_length": length1,
                "padding_seconds": padding1,
                "transcription": transcription1,
                "gt_lufs": y1_lufs,
                "cutoff_freq": freq_1,
                "ff_signal_power_audio1": round(float(ff_signal_power_audio1), 7),
            },
            {
                "file": os.path.basename(file2_path),
                "original_length": length2,
                "padding_seconds": padding2,
                "transcription": transcription2,
                "gt_lufs": y2_lufs,
                "cutoff_freq": freq_2,
                "ff_signal_power_audio2_before_scale": round(
                    float(ff_signal_power_audio2), 7
                ),
                "audios_SIS_scale_db": round(scale_factor_DB, 3),
                "audios_SIS_scale_linear": round(float(linear_scale_factor), 3),
                "audios_SIS_linear_mult_factor": round(float(linear_mult_factor), 3),
            },
        ],
    }
    metadata.append(metadata_entry)


def process_common_voice(
    dataset,
    output_dir: str,
    metadata_file_path: str,
    desired_mixtures_amount: int,
    target_sample_rate: int,
    max_noise_desired_snr: int,
    min_noise_desired_snr: int,
    max_music_ssr: int,
    min_music_ssr: int,
    max_conversation_desired_ssr: int,
    min_conversation_desired_ssr: int,
    rir_directory: str,
    music_directory: str,
    compression: Dict[str, str | bool | int],
    normalize_lufs: bool,
    low_pass_filter_config: Dict[str, bool | int],
    split: str,
):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    # Extract audio paths and transcriptions
    audio_files = dataset["path"]
    transcriptions = dataset["sentence"]

    # Shuffle the audio files and their corresponding transcriptions
    data = list(zip(audio_files, transcriptions))
    # random.shuffle(data)

    # Create overlapping pairs (non-redundant)
    for i in tqdm(
        range(0, desired_mixtures_amount * 2 - 1, 2), desc="Processing Mixtures"
    ):
        # Step by 2 to ensure no file is reused
        file1_path, transcription1 = data[i]
        file2_path, transcription2 = data[i + 1]
        mix_audio(
            file1_path,
            file2_path,
            transcription1,
            transcription2,
            target_sample_rate,
            max_noise_desired_snr,
            min_noise_desired_snr,
            max_music_ssr,
            min_music_ssr,
            max_conversation_desired_ssr,
            min_conversation_desired_ssr,
            output_dir,
            metadata,
            rir_directory=rir_directory,
            music_directory=music_directory,
            compression=compression,
            normalize_lufs=normalize_lufs,
            low_pass_filter_config=low_pass_filter_config,
            split=split,
        )

    # Save metadata as JSON
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    start_time = time.time()  # Start timer
    # load config
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create overlapped test set mixtures.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./src/configs/create_overlapped_test_set_config.json",
        help="Path to the configuration JSON file.",
    )
    args = parser.parse_args()
    # Load config from the provided config_path argument
    config_path = args.config_path
    print(f"config path:{config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    # fmt: off
    max_conversation_desired_ssr = config["signal_to_signal_ratios"]["max_conversation_desired_ssr"]
    min_conversation_desired_ssr = config["signal_to_signal_ratios"]["min_conversation_desired_ssr"]
    max_noise_desired_snr = config["signal_to_signal_ratios"]["max_noise_desired_snr"]
    min_noise_desired_snr = config["signal_to_signal_ratios"]["min_noise_desired_snr"]
    max_music_ssr = config["signal_to_signal_ratios"]["max_music_ssr"]
    min_music_ssr = config["signal_to_signal_ratios"]["min_music_ssr"]
    split = config["dataset_split"]
    # load dataset
    dataset = load_dataset("mozilla-foundation/common_voice_12_0",config["dataset_language"],split=config["dataset_split"],trust_remote_code=True)
    if "desired_mixtures_amount" in config:
        desired_mixtures_amount=config["desired_mixtures_amount"]
    else:
        desired_mixtures_amount =int(len(dataset) / 2) 

    # fmt: on
    # Directories
    # Get the current date and time

    current_datetime = datetime.now()

    # Format the date and time with underscores
    formatted_date = current_datetime.strftime("%d_%m_%Y_%H_%M_%S")
    # fmt: off
    output_dir_name = f"{split}"\
                      f"_{formatted_date}"\
                      f"_{max_conversation_desired_ssr}_{min_conversation_desired_ssr}"\
                      f"_{max_noise_desired_snr}_{min_noise_desired_snr}"\
                      f"_{max_music_ssr}_{min_music_ssr}"
    # fmt: on
    output_directory = os.path.join(
        config["directories"]["main_directory"], output_dir_name
    )
    os.makedirs(output_directory, exist_ok=True)
    # save config file
    file_path = os.path.join(output_directory, "config.json")
    with open(file_path, "w") as f:
        json.dump(config, f, indent=4)
    metadata_file_path = os.path.join(output_directory, "metadata.json")
    # Process dataset
    process_common_voice(
        dataset=dataset,
        output_dir=output_directory,
        metadata_file_path=metadata_file_path,
        desired_mixtures_amount=desired_mixtures_amount,
        target_sample_rate=config["target_sample_rate"],
        max_noise_desired_snr=max_noise_desired_snr,
        min_noise_desired_snr=min_noise_desired_snr,
        max_conversation_desired_ssr=max_conversation_desired_ssr,
        min_conversation_desired_ssr=min_conversation_desired_ssr,
        max_music_ssr=max_music_ssr,
        min_music_ssr=min_music_ssr,
        rir_directory=config["directories"]["rir_directory"],
        music_directory=config["directories"]["music_directory"],
        compression=config["compression"],
        normalize_lufs=config["normalize_lufs"],
        low_pass_filter_config=config["low_pass_filter"],
        split=split,
    )
    # Calculate and print execution time
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60
    elapsed_hours = elapsed_minutes / 60

    print(f"\nTotal Run Time: {elapsed_seconds:.2f} seconds")
    print(f"Total Run Time: {elapsed_minutes:.2f} minutes")
    print(f"Total Run Time: {elapsed_hours:.2f} hours")
