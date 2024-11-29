
## Environment setup

```
Use Python 3.12.6
pip install -r ./requirements.txt
```

## config
create config.json file in src and run program
```
{
    "desired_mixtures_amount": 3, # Output mixtures amount
    "target_sample_rate": 8000, # Datasets sample rate. Resamples if needed
    "signal_to_signal_ratios": {
        "max_noise_desired_snr": 12, # DB scale for added white noise
        "min_noise_desired_snr": -3,
        "max_conversation_desired_ssr": 9, # DB scale signal to signal raio between audios
        "min_conversation_desired_ssr": -9,
        "max_music_ssr": -3, # DB scale signal to signal raio between mixture to added music
        "min_music_ssr": -9
    },
    "directories": {
        "main_directory": "path to output dir", 
        "rir_directory": "path to rir dir",
        "music_directory": "path to additional music dir"
    }
}

```