
## Environment setup

```
Use Python 3.12.6
pip install -r ./requirements.txt
```

## config
create config.json file in src and run program
```
{
    "target_sample_rate": 8000,# Datasets sample rate. Resamples if needed
    "signal_to_signal_ratios": {
        "max_noise_desired_snr": 20, # DB scale for added white noise
        "min_noise_desired_snr": 10,
        "max_conversation_desired_ssr": 8, # DB scale signal to signal raio between audios
        "min_conversation_desired_ssr": 4,
        "max_music_ssr": 15, # DB scale signal to signal raio between mixture to added music
        "min_music_ssr": 10
    },
    "compression":{ # apply_compression= true will save compressed wav files. compressions configruations should be provided
        "apply_compression":true, 
        "min_bitrate":8000,
        "max_bitrate":16000,
        "compression_config":{
            "bitrate": 8000,
            "bit_depth": 16,
            "sample_rate": 8000,
            "opus_channels_number": 1,
            "opus_complexity": 10,
            "opus_enable_dtx": 0,
            "opus_sample_rate_factor": 8000,
            "frame_duration_ms": 20}},
    "directories": {
        "main_directory": "path to output dir"
        "rir_directory": "path to rir dir"
        "music_directory": "path to additional music dir"
    },
    "dataset_split":str # huggingface dataset split
    "dataset_language":str# hugginface dataset language
    "desired_mixtures_amount":int  # Output mixtures amount
    "normalize_lufs":bool, # wether to normalize the convoloved signal to a level of loudness - hard coded level
    "low_pass_filter":{"apply_low_pass_filter":bool, "cutoff_freq":int} # low pass filter coniguration - optional 
}
```
