from datasets import load_dataset
import soundfile as sf
import librosa
import numpy as np
import scipy.signal
TARGET_SAMPLE_RATE = 8000

dataset = load_dataset("mozilla-foundation/common_voice_12_0", "ar", split="test",trust_remote_code=True)
# Get the path to the first audio file
audio_path = dataset[0]["path"]

# Load the audio file
audio_waveform, wav_sample_rate = sf.read(audio_path)
print(f"original sample_rate:{wav_sample_rate}")

# resample wave form
if wav_sample_rate != TARGET_SAMPLE_RATE:
    audio_waveform = librosa.resample(audio_waveform, orig_sr=wav_sample_rate, target_sr=TARGET_SAMPLE_RATE)
# Load the RIR from the subfolder
rir_path = "C:/Users/arifr/SS_Dataset_preparation/MIT_RIR/h001_Bedroom_65txts.wav"  # Replace with the actual path to your RIR file
rir_waveform, rir_sample_rate = sf.read(rir_path)
if rir_sample_rate != TARGET_SAMPLE_RATE:
    rir_waveform = librosa.resample(rir_waveform, orig_sr=rir_sample_rate, target_sr=TARGET_SAMPLE_RATE)

# Normalize the RIR to avoid amplifying the audio excessively
rir_waveform = rir_waveform / np.max(np.abs(rir_waveform))

# Apply the RIR using convolution
convolved_waveform = scipy.signal.convolve(audio_waveform, rir_waveform, mode="full")

# Normalize the output to prevent clipping
convolved_waveform = convolved_waveform / np.max(np.abs(convolved_waveform))

# Save the output audio to a file
convolved_output_path = "./convolved_audio.wav"
uncolvolved_output_path = "./unconvolved_audio.wav"
sf.write(convolved_output_path, convolved_waveform, TARGET_SAMPLE_RATE)
sf.write(uncolvolved_output_path,audio_waveform, TARGET_SAMPLE_RATE)
