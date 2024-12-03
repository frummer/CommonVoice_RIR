import json
import os
import random
from datetime import datetime
from typing import Dict
from uuid import uuid4

import librosa
import numpy as np
import scipy.signal
import soundfile as sf
from datasets import load_dataset, load_from_disk

from utils.opus_codec import encode_decode_opus


def normalize_audio(audio):
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    return audio


def add_music_to_mixed_file(music, wav_file, music_scale: int):
    if len(music) > len(wav_file):
        music = music[: len(wav_file)]  # Truncate
    else:
        padding = np.zeros(len(wav_file) - len(music))
        music = np.concatenate((music, padding))  # Pad with zeros

    # Sum the two signals
    combined_audio = wav_file + music * music_scale
    # Normalize the combined audio to avoid clipping
    combined_audio = normalize_audio(combined_audio)
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

    # Generate white Gaussian noise with the calculated power
    noise = np.sqrt(noise_power) * np.random.normal(0, 1, len(audio))

    # Add noise to the original audio
    noisy_audio = audio + noise
    noisy_audio = normalize_audio(noisy_audio)
    # return noisy audio and also linear snr for metadata saving
    return noisy_audio, snr_linear


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
    convolved_audio = scipy.signal.fftconvolve(audio, rir, mode="valid")
    return convolved_audio


def calc_scale_factor(audio1, audio2, sns_db_scale: int):
    """Calc power of each singal."""
    # Calculate the power of the audio signal
    signal_power_audio1 = np.mean(audio1**2)
    signal_power_audio2 = np.mean(audio2**2)

    # Calculate the desired noise power to achieve the given SNR
    snr_linear = 10 ** (sns_db_scale / 10)
    # snr_linear = power(y1)/(a^2*power(y2))
    a = np.sqrt(signal_power_audio1 / (signal_power_audio2 * snr_linear))
    return a


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
    opus_codec: Dict[str, str | bool],
):
    # Load audio files
    y1, sr1 = librosa.load(file1_path, sr=None)
    y2, sr2 = librosa.load(file2_path, sr=None)

    if sr1 != target_sample_rate:
        y1 = librosa.resample(y1, orig_sr=sr1, target_sr=target_sample_rate)

    if sr2 != target_sample_rate:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=target_sample_rate)

    # Generate unique ID and save mixed audio
    unique_id = str(uuid4())
    subdirectory_path = os.path.join(output_path, unique_id)
    os.makedirs(subdirectory_path, exist_ok=True)

    # Save original files
    original_file1 = os.path.join(subdirectory_path, os.path.basename(file1_path))
    original_file2 = os.path.join(subdirectory_path, os.path.basename(file2_path))
    sf.write(original_file1, y1, target_sample_rate)
    sf.write(original_file2, y2, target_sample_rate)

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

    # apply Room impulse on audio + scaling and save file
    rir, _ = load_random_wav(
        directory=rir_directory, target_sample_rate=target_sample_rate
    )
    ff_audio1 = apply_rir_to_audio(y1, rir)
    ff_audio2 = apply_rir_to_audio(y2, rir)

    # Sample scale factor between audios
    scale_factor_DB = np.random.uniform(
        min_conversation_desired_ssr, max_conversation_desired_ssr
    )
    linear_scale_factor = calc_scale_factor(
        audio1=ff_audio1, audio2=ff_audio2, sns_db_scale=scale_factor_DB
    )

    # Mix the audio
    ff_mixed_audio = ff_audio1 + linear_scale_factor * ff_audio2
    ff_mixed_audio = normalize_audio(ff_mixed_audio)

    # Save far-field audio
    mixed_audio_ff_file = os.path.join(subdirectory_path, f"ff_{unique_id}.wav")
    sf.write(mixed_audio_ff_file, ff_mixed_audio, target_sample_rate)

    # Save encoded far-field audio
    if opus_codec["apply_opus"]:
        mixed_audio_ff_compressed_file = os.path.join(
            subdirectory_path, f"ff_{unique_id}_opus.wav"
        )
        encode_decode_opus(
            input_file_path=mixed_audio_ff_file,
            output_file_path=mixed_audio_ff_compressed_file,
            bit_rate=opus_codec["bitrate"],
        )

    # Make audios noisy
    snr_db = np.random.uniform(min_noise_desired_snr, max_noise_desired_snr)
    noisy_ff_mixed_audio, snr_linear = add_noise_to_match_snr(ff_mixed_audio, snr_db)

    # Save noisy far-field audio
    noisy_ff_mixed_audio_file_path = os.path.join(
        subdirectory_path, f"{unique_id}_noisy.wav"
    )
    sf.write(noisy_ff_mixed_audio_file_path, noisy_ff_mixed_audio, target_sample_rate)
    # Save encoded noisy far-field audio
    # Save encoded far-field audio
    if opus_codec["apply_opus"]:
        noisy_ff_mixed_audio_compressed_file_path = os.path.join(
            subdirectory_path, f"ff_{unique_id}_noisy_opus.wav"
        )
        encode_decode_opus(
            input_file_path=noisy_ff_mixed_audio_file_path,
            output_file_path=noisy_ff_mixed_audio_compressed_file_path,
            bit_rate=opus_codec["bitrate"],
        )

    # add random noise from music directory and save file
    additional_music_wav, addition_music_str = load_random_wav(
        directory=music_directory, target_sample_rate=target_sample_rate
    )
    # Sample scale factor between audios
    music_scale_factor_DB = np.random.uniform(min_music_ssr, max_music_ssr)
    music_linear_scale_factor = calc_scale_factor(
        audio1=y1, audio2=y2, sns_db_scale=music_scale_factor_DB
    )

    noisy_ff_with_music_mixed_audio = add_music_to_mixed_file(
        music=additional_music_wav,
        wav_file=noisy_ff_mixed_audio,
        music_scale=music_linear_scale_factor,
    )

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
    if opus_codec["apply_opus"]:
        noisy_ff_mixed_audio_with_music_compressed_file_path = os.path.join(
            subdirectory_path, f"ff_{unique_id}_noisy_with_music_opus.wav"
        )
        encode_decode_opus(
            input_file_path=mixed_audio_ff_file_with_music_path,
            output_file_path=noisy_ff_mixed_audio_with_music_compressed_file_path,
            bit_rate=opus_codec["bitrate"],
        )
    # save additional music
    additional_music_path = os.path.join(subdirectory_path, addition_music_str)
    sf.write(additional_music_path, additional_music_wav, target_sample_rate)

    # Save metadata
    metadata_entry = {
        "id": unique_id,
        "length_seconds": len(ff_mixed_audio) / target_sample_rate,
        "added_noise_snr_db": round(snr_db, 3),
        "added_noise_snr_linear": round(float(snr_linear), 3),
        "music_2_audio_SIS_scale_db": round(music_scale_factor_DB, 3),
        "music_2_audio_SIS_scale_linear": round(float(music_linear_scale_factor), 3),
        "additional_music": addition_music_str,
        "original_files": [
            {
                "file": os.path.basename(file1_path),
                "original_length": length1,
                "padding_seconds": padding1,
                "transcription": transcription1,
            },
            {
                "file": os.path.basename(file2_path),
                "original_length": length2,
                "padding_seconds": padding2,
                "transcription": transcription2,
                "audios_SIS_scale_db": round(scale_factor_DB, 3),
                "audios_SIS_scale_linear": round(float(linear_scale_factor), 3),
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
    opus_codec: Dict[str, str | bool],
):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    # Extract audio paths and transcriptions
    audio_files = dataset["path"]
    transcriptions = dataset["sentence"]

    # Shuffle the audio files and their corresponding transcriptions
    data = list(zip(audio_files, transcriptions))
    random.shuffle(data)

    # Create overlapping pairs (non-redundant)
    for i in range(0, desired_mixtures_amount * 2 - 1, 2):
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
            opus_codec=opus_codec,
        )

    # Save metadata as JSON
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # load config
    config_path = os.getenv(
        "CONFIG_PATH", "./src/create_overlapped_dataset_config.json"
    )  # Fallback to a default
    with open(config_path, "r") as f:
        config = json.load(f)
    # fmt: off
    max_conversation_desired_ssr = config["signal_to_signal_ratios"]["max_conversation_desired_ssr"]
    min_conversation_desired_ssr = config["signal_to_signal_ratios"]["min_conversation_desired_ssr"]
    max_noise_desired_snr = config["signal_to_signal_ratios"]["max_noise_desired_snr"]
    min_noise_desired_snr = config["signal_to_signal_ratios"]["min_noise_desired_snr"]
    max_music_ssr = config["signal_to_signal_ratios"]["max_music_ssr"]
    min_music_ssr = config["signal_to_signal_ratios"]["min_music_ssr"]
    # load dataset
    dataset = load_dataset("mozilla-foundation/common_voice_12_0","ar",split="test",trust_remote_code=True)
    # fmt: on
    # Directories
    # Get the current date and time
    current_datetime = datetime.now()

    # Format the date and time with underscores
    formatted_date = current_datetime.strftime("%d_%m_%Y_%H_%M_%S")
    # fmt: off
    output_dir_name = f"{formatted_date}"\
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
        desired_mixtures_amount=config["desired_mixtures_amount"],
        target_sample_rate=config["target_sample_rate"],
        max_noise_desired_snr=max_noise_desired_snr,
        min_noise_desired_snr=min_noise_desired_snr,
        max_conversation_desired_ssr=max_conversation_desired_ssr,
        min_conversation_desired_ssr=min_conversation_desired_ssr,
        max_music_ssr=max_music_ssr,
        min_music_ssr=min_music_ssr,
        rir_directory=config["directories"]["rir_directory"],
        music_directory=config["directories"]["music_directory"],
        opus_codec=config["opus_codec"],
    )
