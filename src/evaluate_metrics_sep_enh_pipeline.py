# iterate over metadata.json file

# load relevant files

# calculate metrics

import json
import os

from pesq import pesq

from evaluation_utils import additional_metrics_utils
from evaluation_utils.load_files_utils import (
    load_and_validate_config,
    load_audio_and_resmaple,
    load_metadata,
)


def main():
    """Main function to execute the script."""
    # load config
    config_path = os.getenv(
        "CONFIG_PATH", "./src/evaluate_metrics_sep_enh_pipeline_config.json"
    )
    config = load_and_validate_config(file_path=config_path)
    dataset_dir = config["directories"]["dataset_directory"]
    separatd_audios_dir = config["directories"]["separatd_audios_directory"]
    target_sample_rate = config["target_sample_rate"]
    # Step 1: Load metadata
    metadata = load_metadata(dataset_dir=dataset_dir)

    # Load your audio sample
    # Step 2: Loop over each entry
    for index, entry in enumerate(metadata):
        # Step 3:
        # Step 3.1 load data to ram:
        # load id from entry
        id = entry["id"]
        # load - reference and transcriptions
        # load gt1_name
        gt1_file_name = entry["original_files"][0]["file"]
        # load gt1_audio_file
        gt1_audio_file_path = os.path.join(dataset_dir, id, gt1_file_name)
        # load and resample
        gt1_audio, _resampled1 = load_audio_and_resmaple(
            audio_path=gt1_audio_file_path, target_sample_rate=target_sample_rate
        )
        # load gt2_name
        gt2_file_name = entry["original_files"][1]["file"]
        # load gt2_audio_file
        gt2_audio_file_path = os.path.join(dataset_dir, id, gt2_file_name)
        # load and resample
        gt2_audio, _resampled2 = load_audio_and_resmaple(
            audio_path=gt2_audio_file_path, target_sample_rate=target_sample_rate
        )
        # load - separated audio1
        separated_file_1_path = os.path.join(
            separatd_audios_dir, f"ff_{id}_noisy_with_music_opus_spk1_corrected.wav"
        )
        separated_audio_file_1, _resampled3 = load_audio_and_resmaple(
            audio_path=separated_file_1_path, target_sample_rate=target_sample_rate
        )
        # load - separated audio2
        separated_file_2_path = os.path.join(
            separatd_audios_dir, f"ff_{id}_noisy_with_music_opus_spk2_corrected.wav"
        )
        separated_audio_file_2, _resampled4 = load_audio_and_resmaple(
            audio_path=separated_file_2_path, target_sample_rate=target_sample_rate
        )
        print("here")
        print(f"len(separated_audio_file_2):{len(separated_audio_file_2)}")
        print(f"len(separated_audio_file_1):{len(separated_audio_file_1)}")
        print(f"len(gt2_audio):{len(gt2_audio)}")
        print(f"len(gt1_audio):{len(gt1_audio)}")

        # n1_1 = separated_audio_file_2 - gt2_audio
        # n1_2 = separated_audio_file_1 - gt1_audio
        # n2_1 = separated_audio_file_1 - gt2_audio
        # n2_2 = separated_audio_file_2 - gt1_audio
        print("here")


if __name__ == "__main__":
    main()
