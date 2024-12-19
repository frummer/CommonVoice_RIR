import os

import librosa
import scipy.signal
import soundfile as sf


def apply_rir_to_audio(audio, rir, mode="valid"):
    """Convolve the audio with the RIR."""
    # convolved_audio = scipy.signal.convolve(audio, rir, mode="full")
    convolved_audio = scipy.signal.fftconvolve(audio, rir, mode=mode)
    return convolved_audio


def load_wav_and_resample(wav_path: str, target_sample_rate=8000):
    waveform, wav_sample_rate = librosa.load(wav_path, sr=None)
    if wav_sample_rate != target_sample_rate:
        waveform = librosa.resample(
            waveform, orig_sr=wav_sample_rate, target_sr=target_sample_rate
        )
    return waveform


def write_audio_in_current_dir(audio, mode: str, target_sample_rate: int):
    subdirectory_path = os.path.join(os.getcwd(), mode + ".wav")
    # Save original files
    sf.write(subdirectory_path, audio, target_sample_rate)


if __name__ == "__main__":
    TARGET_SAMPLE_RATE = 8000
    MODES = ["valid", "full", "same"]
    SAMPLE_AUDIO_PATH = "C:\\Users\\arifr\git\\CommonVoice_RIR\\rir_experiments\\common_voice_ar_25175937.mp3"
    RIR_PATH = "C:\\Users\\arifr\\git\\CommonVoice_RIR\\rir_experiments\\h001_Bedroom_65txts.wav"

    # load audio and rir
    audio = load_wav_and_resample(
        wav_path=SAMPLE_AUDIO_PATH, target_sample_rate=TARGET_SAMPLE_RATE
    )
    rir = load_wav_and_resample(
        wav_path=RIR_PATH, target_sample_rate=TARGET_SAMPLE_RATE
    )

    for mode in MODES:
        # apply rir
        output_wav = apply_rir_to_audio(audio=audio, rir=rir, mode=mode)
        # save_output
        write_audio_in_current_dir(
            audio=output_wav, mode=mode, target_sample_rate=TARGET_SAMPLE_RATE
        )
