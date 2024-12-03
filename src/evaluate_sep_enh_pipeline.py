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
        # Step 3: Call transcribe_and_evaluate for each entry
        # transcribe_and_evaluate(entry)
        print(f"index:{index}")


if __name__ == "__main__":
    main()
