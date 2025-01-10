import logging

import torchaudio
import torchaudio.functional as F
from IPython.display import Audio, display
from torchaudio.io import AudioEffector

logging.basicConfig(level=logging.DEBUG)


def play_audio(tensor, sr):
    """Helper to play audio in a notebook (if needed)."""
    return Audio(tensor.squeeze().numpy(), rate=sr)


# 1. Load your audio file

audio_file_path = "C:\\Users\\arifr\\OneDrive\\Desktop\\15_12_2024_21_49_46_7_-5_10_7_10_12\\1c95aafa-707c-4932-8761-c0c24eb0e67f\\common_voice_ar_24060068.mp3"
waveform, sample_rate = torchaudio.load(audio_file_path)
print(f"Loaded waveform shape: {waveform.shape}, sample rate: {sample_rate}")

# 2. (Optional) Resample to 48 kHz (Opus commonly uses 48 kHz for best quality)
target_sr = 8000
if sample_rate != target_sr:
    waveform = F.resample(waveform, sample_rate, target_sr)
    sample_rate = target_sr

print("Original Audio:")
display(play_audio(waveform, sample_rate))

# 4. Encode the waveform to in-memory bytes at ~64 kbps
effect = {"format": "ogg", "encoder": "opus"}
effector = AudioEffector(effect=effect, pad_end=False)
result = effector.apply(waveform, int(sample_rate))
