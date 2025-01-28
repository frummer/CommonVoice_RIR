import os

import numpy as np
import opuslib
import torch
import torchaudio
from opuslib import constants
from opuslib.api import ctl as opuslib_ctl
from opuslib.api import decoder as opuslib_decoder
from opuslib.api import encoder as opuslib_encoder


class BaseBytesEncoderDecoder:
    def __init__(self, bit_depth: int):
        self.bit_depth = bit_depth


class OpusBytesEncoderDecoder(BaseBytesEncoderDecoder):
    def __init__(self, bitrate: int, bit_depth: int):
        super().__init__(bit_depth)
        self.bitrate = bitrate

    def reset_state(self, sample_rate: int, config: dict):
        self.config = config  # Save config for potential later use
        sample_rate_for_opus = int(
            config["opus_sample_rate_factor"]
            * np.ceil(sample_rate / config["opus_sample_rate_factor"])
        )

        enc_state = opuslib_encoder.create_state(
            fs=sample_rate_for_opus,
            channels=config["opus_channels_number"],
            application=opuslib.APPLICATION_AUDIO,
        )

        opuslib_encoder.encoder_ctl(enc_state, opuslib_ctl.set_bitrate, self.bitrate)
        opuslib_encoder.encoder_ctl(
            enc_state, opuslib_ctl.set_complexity, config["opus_complexity"]
        )
        opuslib_encoder.encoder_ctl(
            enc_state, opuslib_ctl.set_lsb_depth, self.bit_depth
        )
        opuslib_encoder.encoder_ctl(
            enc_state, opuslib_ctl.set_dtx, config["opus_enable_dtx"]
        )
        opuslib_encoder.encoder_ctl(
            enc_state, opuslib_ctl.set_max_bandwidth, constants.BANDWIDTH_FULLBAND
        )

        self.encoder = opuslib.classes.Encoder(
            fs=sample_rate_for_opus,
            channels=config["opus_channels_number"],
            application=opuslib.APPLICATION_AUDIO,
        )
        self.encoder.encoder_state = enc_state
        opuslib_encoder.encoder_ctl(
            self.encoder.encoder_state, opuslib_ctl.set_bitrate, self.bitrate
        )

        dec_state = opuslib_decoder.create_state(
            fs=sample_rate_for_opus, channels=config["opus_channels_number"]
        )

        self.decoder = opuslib.classes.Decoder(
            fs=sample_rate_for_opus, channels=config["opus_channels_number"]
        )
        self.decoder.decoder_state = dec_state

    def encode(self, input_data: bytes, frame_size: int) -> bytes:
        return self.encoder.encode(pcm_data=input_data, frame_size=frame_size)

    def decode(self, input_data: bytes, frame_size: int) -> bytes:
        return self.decoder.decode(opus_data=input_data, frame_size=frame_size)


# Helper functions
def preprocess_waveform(waveform, sample_rate, target_sample_rate, config):
    """
    Preprocess a waveform by resampling it, dividing it into frames,
    and preparing PCM byte data for Opus encoding.

    Args:
        waveform (Tensor | np.ndarray): Input waveform (channels x samples or samples).
        sample_rate (int): Original sample rate.
        target_sample_rate (int): Target sample rate for preprocessing.
        config (dict): Configuration dictionary with Opus settings.

    Returns:
        list[bytes]: List of PCM byte chunks ready for encoding.
    """
    # Convert NumPy array to Torch tensor if needed
    if isinstance(waveform, np.ndarray):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    # Ensure waveform is in the correct shape (channels x samples)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension if missing

    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )
        waveform = resampler(waveform)

    # Scale and convert to int16 PCM bytes
    waveform = waveform * (2**15 - 1)
    waveform = waveform.to(dtype=torch.int16)

    # Flatten the waveform for PCM byte preparation
    pcm_data = waveform.numpy().tobytes()

    # Calculate frame size
    frame_size = (config["sample_rate"] // 1000) * config["frame_duration_ms"]

    # Divide PCM data into frames
    total_samples = len(pcm_data) // (
        2 * config["opus_channels_number"]
    )  # 2 bytes per sample
    num_frames = total_samples // frame_size

    pcm_frames = []
    for i in range(num_frames):
        start = i * frame_size * 2 * config["opus_channels_number"]
        end = start + frame_size * 2 * config["opus_channels_number"]
        chunk = pcm_data[start:end]

        if len(chunk) == frame_size * 2 * config["opus_channels_number"]:
            pcm_frames.append(chunk)

    return pcm_frames


def postprocess_waveform(decoded_pcm, channels):
    """
    Postprocess decoded PCM bytes into a waveform.

    Args:
        decoded_pcm (bytes): Decoded PCM byte data.
        channels (int): Number of audio channels.

    Returns:
        Tensor: Decoded waveform (channels x samples).
    """
    decoded_pcm = bytearray(decoded_pcm)  # Ensure the buffer is writable
    waveform = torch.frombuffer(decoded_pcm, dtype=torch.int16).view(channels, -1)

    # Transpose the waveform to match (samples x channels) format for saving
    waveform = waveform.transpose(0, 1).numpy()
    return waveform


if __name__ == "__main__":
    # Configuration
    config = {
        "bitrate": 6000,
        "bit_depth": 16,
        "sample_rate": 8000,
        "opus_channels_number": 1,
        "opus_complexity": 10,
        "opus_enable_dtx": 0,
        "opus_sample_rate_factor": 8000,
        "frame_duration_ms": 20,  # Frame duration in milliseconds
        "audio_dir": "/app/files_to_compress",
        "decoded_output_dir": "/app/compress_decompress_files",
    }

    # Initialize encoder-decoder
    encoder_decoder = OpusBytesEncoderDecoder(
        bitrate=config["bitrate"], bit_depth=config["bit_depth"]
    )
    encoder_decoder.reset_state(sample_rate=config["sample_rate"], config=config)

    # Create output directory if it doesn't exist
    os.makedirs(config["decoded_output_dir"], exist_ok=True)

    # Process all audio files in the directory
    for filename in os.listdir(config["audio_dir"]):
        if filename.endswith(".wav") or filename.endswith(".flac"):
            audio_path = os.path.join(config["audio_dir"], filename)
            output_path = os.path.join(
                config["decoded_output_dir"], f"decoded_{filename}"
            )

            try:
                # Load audio and preprocess
                waveform, sample_rate = torchaudio.load(audio_path)
                pcm_frames = preprocess_waveform(
                    waveform, sample_rate, config["sample_rate"], config
                )

                # Encode and decode the audio in frames
                decoded_pcm = []
                frame_size = (config["sample_rate"] // 1000) * config[
                    "frame_duration_ms"
                ]

                for chunk in pcm_frames:
                    encoded_data = encoder_decoder.encode(
                        input_data=chunk, frame_size=frame_size
                    )
                    decoded_frame = encoder_decoder.decode(
                        input_data=encoded_data, frame_size=frame_size
                    )
                    decoded_pcm.append(decoded_frame)

                # Combine and postprocess the decoded audio
                decoded_pcm = b"".join(decoded_pcm)
                decoded_waveform = postprocess_waveform(
                    decoded_pcm, config["opus_channels_number"]
                )

                # Save decoded waveform to file
                torchaudio.save(
                    output_path, torch.tensor(decoded_waveform).T, config["sample_rate"]
                )

                print(f"Decoded {filename} to {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
