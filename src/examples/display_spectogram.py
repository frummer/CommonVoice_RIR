import librosa
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from matplotlib import pyplot as plt


def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    # print(waveform)
    # print()


def plot_spectrogram(
    spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None, save_path=None
):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show(block=True)


n_fft = 1024
win_length = None
hop_length = 512

# define transformation
spectrogram = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
)

if __name__ == "__main__":
    input_file = "C:\\Users\\arifr\\git\\CommonVoice_RIR\\filtered_samples\\00001_filtered_audio.wav"
    waveform, sample_rate = torchaudio.load(input_file)

    # Load the audio file with librosa
    # Perform transformation
    spec = spectrogram(waveform)

    print_stats(spec)
    plot_spectrogram(spec[0], title="spectogram", save_path="./")
