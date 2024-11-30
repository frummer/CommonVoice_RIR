import os

import librosa
import soundfile as sf


def resample_wav(input_file, output_file, target_rate=16000):
    try:
        # Load the audio file with librosa
        data, original_rate = librosa.load(input_file, sr=None)
        print(f"Original Sample Rate: {original_rate} Hz")

        # Resample the audio to the target sample rate
        resampled_data = librosa.resample(
            data, orig_sr=original_rate, target_sr=target_rate
        )
        print(f"Resampling to {target_rate} Hz...")

        # Save the resampled audio to the output file
        sf.write(output_file, resampled_data, target_rate)
        print(f"Resampled file saved to: {output_file}")
        print(f"New Sample Rate: {target_rate} Hz")

    except Exception as e:
        print(f"Error processing file: {e}")


# Example usage
input_wav = "input path"  # Path to the input WAV file
output_wav = "output_path"  # Path to save the output WAV file
target_sample_rate = 16000  # Desired sample rate

if os.path.exists(input_wav):
    resample_wav(input_wav, output_wav, target_sample_rate)
else:
    print(f"Input file '{input_wav}' not found.")
