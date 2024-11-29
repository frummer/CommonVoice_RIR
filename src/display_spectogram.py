import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
input_file = "./src/common_voice_ar_19065197_decoded.wav"
y, sr = librosa.load(input_file, sr=None)  # Load with the original sample rate

# Generate the spectrogram
S = librosa.feature.melspectrogram(
    y=y, sr=sr, n_mels=128, fmax=sr / 2

# Convert to log scale (dB)
S_dB = librosa.power_to_db(S, ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(
    S_dB, sr=sr, x_axis="time", y_axis="mel", fmax=sr / 2, cmap="viridis"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Mel-Frequency Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
plt.show()
