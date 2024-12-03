import json
import os

from evaluation_utls.load_files_utils import load_and_validate_config, load_metadata


def main():
    """Main function to execute the script."""
    # load config
    config_path = os.getenv(
        "CONFIG_PATH", "./src/evaluate_sep_enh_pipeline_config.json"
    )
    config = load_and_validate_config(file_path=config_path)
    # Step 1: Load metadata
    metadata = load_metadata(dataset_dir=config["directories"]["dataset_directory"])

    # Step 2: Loop over each entry
    for index, entry in enumerate(metadata):
        # Step 3:
        # Step 3.1 load data to ram:
        # load id from entry
        # load - reference and transcriptions
        # load gt1_name
        # load gt1_audio_file
        # load gt1_transcription
        # load gt2_name
        # load gt2_audio_file
        # load gt2_transcription
        # load - separated audio1
        # load - separated audio2

        # Step 3.2 transcribe all audios:
        # transcribe gt1
        # transcribe gt2
        # transcribe gt1_audio_file
        # transcribe gt2_audio_file

        # Step 3.3 calculate WER:
        # transcribe gt1 vs transcription_g1
        # transcribe gt2 vs transcription_g2
        # transcribe gt1_audio_file vs transcription_g1
        # transcribe gt1_audio_file vs transcription_g2
        # transcribe gt2_audio_file vs transcription_g1
        # transcribe gt2_audio_file vs transcription_g2

        print(f"index:{index}")
        break


if __name__ == "__main__":
    main()
