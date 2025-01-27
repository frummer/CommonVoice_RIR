import os

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio


def lufs_norm(data, sr, norm=-6):
    """
    Normalize the audio signal to a target LUFS value.

    Args:
        data (numpy.ndarray): Input audio signal.
        sr (int): Sample rate of the audio.
        norm (float): Target LUFS value.

    Returns:
        tuple: Normalized audio and gain applied.
    """
    block_size = 0.4 if len(data) / sr >= 0.4 else len(data) / sr
    meter = pyln.Meter(rate=sr, block_size=block_size)
    loudness = meter.integrated_loudness(data)

    if np.isinf(loudness):
        loudness = -40  # Handle infinite loudness cases

    norm_data = pyln.normalize.loudness(data, loudness, norm)
    gain = np.sum(norm_data) / np.sum(data) if np.sum(data) != 0 else 0.0

    return norm_data, gain


def calculate_lufs(audio, sr):
    """
    Calculate the LUFS (integrated loudness) of an audio signal.

    Args:
        audio (numpy.ndarray): Input audio signal.
        sr (int): Sample rate of the audio.

    Returns:
        float: Integrated loudness in LUFS.
    """
    block_size = 0.4 if len(audio) / sr >= 0.4 else len(audio) / sr
    meter = pyln.Meter(rate=sr, block_size=block_size)
    loudness = meter.integrated_loudness(audio)
    return loudness


def get_lufs_norm_audio(audio, sr=16000, lufs=-6):
    """
    Normalize audio to a random LUFS value in the range [lufs-2, lufs+2].

    Args:
        audio (numpy.ndarray): Input audio signal.
        sr (int): Sample rate of the audio.
        lufs (float): Target LUFS value.

    Returns:
        tuple: Normalized audio and gain applied.
    """
    class_lufs = np.random.uniform(lufs - 2, lufs + 2)
    data_norm, gain = lufs_norm(data=audio, sr=sr, norm=class_lufs)
    return data_norm, gain


def normalize_audio_files(input_dir, output_dir, target_lufs_list, sample_rate=16000):
    """
    Normalize all audio files in a directory to a list of target LUFS values.

    Args:
        input_dir (str): Path to the input directory containing audio files.
        output_dir (str): Path to the output directory to save normalized files.
        target_lufs_list (list): List of target LUFS values for normalization.
        sample_rate (int): Sample rate to resample the audio if needed.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            input_path = os.path.join(input_dir, file_name)

            # Load audio
            waveform, sr = torchaudio.load(input_path)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if necessary
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=sample_rate
                )
                waveform = resampler(waveform)

            # Normalize audio for each target LUFS value
            audio_np = waveform.squeeze().numpy()
            for target_lufs in target_lufs_list:
                normalized_audio, gain = get_lufs_norm_audio(
                    audio_np, sr=sample_rate, lufs=target_lufs
                )

                # Save normalized audio
                output_file_name = (
                    f"{os.path.splitext(file_name)[0]}_lufs_{target_lufs}.wav"
                )
                output_path = os.path.join(output_dir, output_file_name)
                normalized_waveform = torch.tensor(normalized_audio).unsqueeze(0)
                torchaudio.save(output_path, normalized_waveform, sample_rate)

                print(
                    f"Normalized {file_name} to {target_lufs} LUFS: Gain applied = {gain:.2f}"
                )


# Example usage
if __name__ == "__main__":
    input_directory = "C:\\Users\\arifr\\git\\CommonVoice_RIR\\sources_samples"
    output_directory = "C:\\Users\\arifr\\git\\CommonVoice_RIR\\scaled_lufs_samples"
    target_lufs_values = [-17, -24, -29]  # List of target LUFS values

    normalize_audio_files(
        input_directory, output_directory, target_lufs_list=target_lufs_values
    )
