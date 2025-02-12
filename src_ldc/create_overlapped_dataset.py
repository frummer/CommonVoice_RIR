import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.create_overlapped_dataset import mix_audio



def process_ldc_dataset(
    csv_path: str,
    output_dir: str,
    metadata_file_path: str,
    target_sample_rate: int,
    max_noise_desired_snr: int,
    min_noise_desired_snr: int,
    max_music_ssr: int,
    min_music_ssr: int,
    max_conversation_desired_ssr: int,
    min_conversation_desired_ssr: int,
    rir_directory: str,
    music_directory: str,
    compression: Dict[str, bool | int],
    normalize_lufs: bool,
    low_pass_filter_config: Dict[str, bool | int],
    split: str,
):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    # Load the dataset
    df = pd.read_csv(csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing LDC Mixtures"):
        file1_path = row["Utterance_1"]
        file2_path = row["Utterance_2"]

        # Ensure files exist before mixing
        if not os.path.exists(file1_path) or not os.path.exists(file2_path):
            print(f"Skipping {file1_path} or {file2_path}, file not found.")
            continue

        mix_audio(
            file1_path=file1_path,
            file2_path=file2_path,
            transcription1="",  # Empty transcription
            transcription2="",  # Empty transcription
            target_sample_rate=target_sample_rate,
            max_noise_desired_snr=max_noise_desired_snr,
            min_noise_desired_snr=min_noise_desired_snr,
            max_music_ssr=max_music_ssr,
            min_music_ssr=min_music_ssr,
            max_conversation_desired_ssr=max_conversation_desired_ssr,
            min_conversation_desired_ssr=min_conversation_desired_ssr,
            output_path=output_dir,
            metadata=metadata,
            rir_directory=rir_directory,
            music_directory=music_directory,
            compression=compression,
            normalize_lufs=normalize_lufs,
            low_pass_filter_config=low_pass_filter_config,
            split=split,
        )

    # Save metadata as JSON
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"Processing complete. Metadata saved at {metadata_file_path}")


if __name__ == "__main__":
    start_time = time.time()  # Start timer

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create overlapped test set mixtures.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./src/configs/create_ldc_overlapped_dataset_set_config.json",
        help="Path to the configuration JSON file.",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = args.config_path
    print(f"config path:{config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    max_conversation_desired_ssr = config["signal_to_signal_ratios"][
        "max_conversation_desired_ssr"
    ]
    min_conversation_desired_ssr = config["signal_to_signal_ratios"][
        "min_conversation_desired_ssr"
    ]
    max_noise_desired_snr = config["signal_to_signal_ratios"]["max_noise_desired_snr"]
    min_noise_desired_snr = config["signal_to_signal_ratios"]["min_noise_desired_snr"]
    max_music_ssr = config["signal_to_signal_ratios"]["max_music_ssr"]
    min_music_ssr = config["signal_to_signal_ratios"]["min_music_ssr"]
    split = config["dataset_split"]

    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%d_%m_%Y_%H_%M_%S")

    output_dir_name = f"{split}_{formatted_date}_{max_conversation_desired_ssr}_{min_conversation_desired_ssr}_{max_noise_desired_snr}_{min_noise_desired_snr}_{max_music_ssr}_{min_music_ssr}"
    output_directory = os.path.join(
        config["directories"]["main_directory"], output_dir_name
    )
    os.makedirs(output_directory, exist_ok=True)

    file_path = os.path.join(output_directory, "config.json")
    with open(file_path, "w") as f:
        json.dump(config, f, indent=4)

    metadata_file_path = os.path.join(output_directory, "metadata.json")

    process_ldc_dataset(
        csv_path=config["csv_path"],
        output_dir=output_directory,
        metadata_file_path=metadata_file_path,
        target_sample_rate=config["target_sample_rate"],
        max_noise_desired_snr=max_noise_desired_snr,
        min_noise_desired_snr=min_noise_desired_snr,
        max_music_ssr=max_music_ssr,
        min_music_ssr=min_music_ssr,
        max_conversation_desired_ssr=max_conversation_desired_ssr,
        min_conversation_desired_ssr=min_conversation_desired_ssr,
        rir_directory=config["directories"]["rir_directory"],
        music_directory=config["directories"]["music_directory"],
        compression=config["compression"],
        normalize_lufs=config["normalize_lufs"],
        low_pass_filter_config=config["low_pass_filter"],
        split=split,
    )

    # Calculate and print execution time
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60
    elapsed_hours = elapsed_minutes / 60

    print(f"\nTotal Run Time: {elapsed_seconds:.2f} seconds")
    print(f"Total Run Time: {elapsed_minutes:.2f} minutes")
    print(f"Total Run Time: {elapsed_hours:.2f} hours")
