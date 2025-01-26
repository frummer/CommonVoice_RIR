import os

import torch
import torchaudio
import torchaudio.functional as F


def lowpass_filter(waveform, sample_rate, cutoff_freq):
    return F.lowpass_biquad(waveform, sample_rate, cutoff_freq)


def highpass_filter(waveform, sample_rate, cutoff_freq):
    return F.highpass_biquad(waveform, sample_rate, cutoff_freq)


def bandpass_filter(waveform, sample_rate, center_freq, Q):
    return F.bandpass_biquad(waveform, sample_rate, center_freq, Q)


def bandreject_filter(waveform, sample_rate, center_freq, Q):
    return F.bandreject_biquad(waveform, sample_rate, center_freq, Q)


def equalizer_filter(waveform, sample_rate, center_freq, gain, Q):
    return F.equalizer_biquad(waveform, sample_rate, center_freq, gain, Q)


def apply_filter(config):
    """
    Apply a specific filter based on the config dictionary.

    Parameters:
        config (dict): Configuration dictionary containing:
            - input (str): Path to the input audio file.
            - output (str): Path to save the filtered audio file.
            - filter (str): Name of the filter to apply.
            - parameters (dict): Parameters specific to the filter.
            - gpu (bool): Whether to use GPU for processing.

    Returns:
        None
    """
    # Load the audio
    waveform, sample_rate = torchaudio.load(config["input"])

    # Move to GPU if specified and available
    if config.get("gpu", False) and torch.cuda.is_available():
        waveform = waveform.to("cuda")

    # Apply the specified filter
    filter_name = config["filter"]
    params = config.get("parameters", {})

    if filter_name == "lowpass_biquad":
        filtered_waveform = lowpass_filter(waveform, sample_rate, **params)
    elif filter_name == "highpass_biquad":
        filtered_waveform = highpass_filter(waveform, sample_rate, **params)
    elif filter_name == "bandpass_biquad":
        filtered_waveform = bandpass_filter(waveform, sample_rate, **params)
    elif filter_name == "bandreject_biquad":
        filtered_waveform = bandreject_filter(waveform, sample_rate, **params)
    elif filter_name == "equalizer_biquad":
        filtered_waveform = equalizer_filter(waveform, sample_rate, **params)
    else:
        raise ValueError(f"Filter {filter_name} is not supported.")

    # Move back to CPU if on GPU
    if config.get("gpu", False) and torch.cuda.is_available():
        filtered_waveform = filtered_waveform.to("cpu")

    # Save the output
    torchaudio.save(config["output"], filtered_waveform, sample_rate)
    print(f"Filtered audio saved to {config['output']}")


if __name__ == "__main__":
    # Example configuration
    config = {
        "input": "C:\\Users\\arifr\\git\\CommonVoice_RIR\\sources_samples\\00001.wav",
        "output": "C:\\Users\\arifr\\git\\CommonVoice_RIR\\filtered_samples\\00001_filtered_audio.wav",
        "filter": "lowpass_biquad",
        "parameters": {"cutoff_freq": 1000},
        "gpu": True,
    }

    apply_filter(config)
