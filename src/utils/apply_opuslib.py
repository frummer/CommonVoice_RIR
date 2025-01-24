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

    def reset_state(self, sample_rate: int):
        sample_rate_for_opus = int(
            OPUS_SAMPLE_RATE_FACTOR * np.ceil(sample_rate / OPUS_SAMPLE_RATE_FACTOR)
        )

        enc_state = opuslib_encoder.create_state(
            fs=sample_rate_for_opus,
            channels=OPUS_CHANNELS_NUMBER,
            application=opuslib.APPLICATION_AUDIO,
        )

        opuslib_encoder.encoder_ctl(enc_state, opuslib_ctl.set_bitrate, self.bitrate)
        opuslib_encoder.encoder_ctl(
            enc_state, opuslib_ctl.set_complexity, OPUS_COMPLEXITY
        )
        opuslib_encoder.encoder_ctl(
            enc_state, opuslib_ctl.set_lsb_depth, self.bit_depth
        )
        opuslib_encoder.encoder_ctl(enc_state, opuslib_ctl.set_dtx, OPUS_ENABLE_DTX)
        opuslib_encoder.encoder_ctl(
            enc_state, opuslib_ctl.set_max_bandwidth, constants.BANDWIDTH_FULLBAND
        )

        self.encoder = opuslib.classes.Encoder(
            fs=sample_rate_for_opus,
            channels=OPUS_CHANNELS_NUMBER,
            application=opuslib.APPLICATION_AUDIO,
        )
        self.encoder.encoder_state = enc_state
        opuslib_encoder.encoder_ctl(
            self.encoder.encoder_state, opuslib_ctl.set_bitrate, self.bitrate
        )

        dec_state = opuslib_decoder.create_state(
            fs=sample_rate_for_opus, channels=OPUS_CHANNELS_NUMBER
        )

        self.decoder = opuslib.classes.Decoder(
            fs=sample_rate_for_opus, channels=OPUS_CHANNELS_NUMBER
        )
        self.decoder.decoder_state = dec_state

    def encode(self, input_data: bytes, frame_size: int) -> bytes:
        return self.encoder.encode(pcm_data=input_data, frame_size=frame_size)

    def decode(self, input_data: bytes, frame_size: int) -> bytes:
        return self.decoder.decode(opus_data=input_data, frame_size=frame_size)


if __name__ == "__main__":
    # Configuration
    config = {
        "bitrate": 64000,  # Example bitrate in bits per second
        "bit_depth": 16,  # Example bit depth
        "sample_rate": 48000,  # Example sample rate
        "opus_channels_number": 1,  # Typically 1 for mono or 2 for stereo
        "opus_complexity": 10,  # Complexity level (0-10)
        "opus_enable_dtx": 0,  # Discontinuous Transmission: 1 to enable, 0 to disable
        "opus_sample_rate_factor": 48000,  # Common Opus sample rate factor
        "audio_dir": "/app/scaled_lufs_samples",  # Directory containing audio files to process
        "decoded_output_dir": "/app/decompressed_samples",  # Directory to save the decoded audio files
    }

    # Assign constants from config
    OPUS_CHANNELS_NUMBER = config["opus_channels_number"]
    OPUS_COMPLEXITY = config["opus_complexity"]
    OPUS_ENABLE_DTX = config["opus_enable_dtx"]
    OPUS_SAMPLE_RATE_FACTOR = config["opus_sample_rate_factor"]

    encoder_decoder = OpusBytesEncoderDecoder(
        bitrate=config["bitrate"], bit_depth=config["bit_depth"]
    )
    encoder_decoder.reset_state(sample_rate=config["sample_rate"])

    # Create output directory if it doesn't exist
    os.makedirs(config["decoded_output_dir"], exist_ok=True)

    # Process all audio files in the specified directory
    for filename in os.listdir(config["audio_dir"]):
        if filename.endswith(".wav") or filename.endswith(".flac"):
            audio_path = os.path.join(config["audio_dir"], filename)
            output_path = os.path.join(
                config["decoded_output_dir"], f"decoded_{filename}"
            )

            # Read audio data using torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Debug: Check channels in the input waveform
            num_channels = waveform.size(0)
            print(f"Processing {filename} with {num_channels} channel(s).")

            # Resample if necessary
            if sample_rate != config["sample_rate"]:
                print(
                    f"Resampling {filename} from {sample_rate} Hz to {config['sample_rate']} Hz."
                )
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=config["sample_rate"]
                )
                waveform = resampler(waveform)

            # Scale waveform to int16 and convert to bytes
            waveform = waveform * (2**15 - 1)
            waveform = waveform.to(dtype=torch.int16)
            input_pcm_data = waveform.numpy().tobytes()

            # Calculate frame size (20 ms frame for 48 kHz sample rate)
            frame_duration_ms = 20  # Frame duration in milliseconds
            frame_size = (config["sample_rate"] // 1000) * frame_duration_ms

            # Debugging: Validate frame size and input data
            print(f"Frame size (samples per channel): {frame_size}")
            print(f"Input PCM data length: {len(input_pcm_data)}")
            print(f"Expected chunk size: {frame_size * 2 * OPUS_CHANNELS_NUMBER}")

            # Process entire input PCM data in chunks of `frame_size`
            total_samples = len(input_pcm_data) // (2 * OPUS_CHANNELS_NUMBER)
            num_frames = total_samples // frame_size

            decoded_pcm = []
            for i in range(num_frames):
                start = i * frame_size * 2 * OPUS_CHANNELS_NUMBER
                end = start + frame_size * 2 * OPUS_CHANNELS_NUMBER
                chunk = input_pcm_data[start:end]

                # Validate chunk size
                if len(chunk) != frame_size * 2 * OPUS_CHANNELS_NUMBER:
                    print(f"Invalid chunk size: {len(chunk)}. Skipping this chunk.")
                    continue

                try:
                    # Encode and decode
                    encoded_data = encoder_decoder.encode(
                        input_data=chunk, frame_size=frame_size
                    )
                    decoded_frame = encoder_decoder.decode(
                        input_data=encoded_data, frame_size=frame_size
                    )
                    decoded_pcm.append(decoded_frame)
                except opuslib.OpusError as e:
                    print(f"OpusError during encoding/decoding: {e}")
                    print(f"Failed on frame {i} with chunk size {len(chunk)}.")
                    break

            # Combine all decoded frames
            decoded_pcm = b"".join(decoded_pcm)

            # Debugging: Print lengths
            print(f"Original PCM length: {len(input_pcm_data)}")
            print(f"Decoded PCM length: {len(decoded_pcm)}")

            # Handle PyTorch warning: Copy the buffer to make it writable
            decoded_pcm = bytearray(decoded_pcm)
            decoded_audio = torch.frombuffer(decoded_pcm, dtype=torch.int16).view(
                OPUS_CHANNELS_NUMBER, -1
            )

            # Save decoded data to the specified output path
            torchaudio.save(
                output_path, decoded_audio, config["sample_rate"], format="wav"
            )

            print(f"Processed {filename}: Decoded data saved to {output_path}")
