#!/usr/bin/env python3

import glob
import json
import os
import random
import shutil


def combine_and_split(
    dir1: str,
    dir2: str,
    combined_dir: str,
    holdout_dir: str,
    subfolders=("mixture", "source1", "source2"),
    fraction=0.2,
):
    """
    Combine wav files from dir1 and dir2 into 'combined_dir', then
    move 'fraction' of files (randomly selected) from 'mixture' into 'holdout_dir',
    along with matching filenames from other subfolders.

    Args:
        dir1 (str): Path to the first directory (has subfolders mixture, source1, source2).
        dir2 (str): Path to the second directory (same structure).
        combined_dir (str): Where to place the combined wavs.
        holdout_dir (str): Where to place the holdout wavs.
        subfolders (tuple): Subfolder names to handle (mixture, source1, source2).
        fraction (float): Fraction of mixture files to move to holdout.
    """
    # 1. Create subfolders in combined and holdout
    for subf in subfolders:
        os.makedirs(os.path.join(combined_dir, subf), exist_ok=True)
        os.makedirs(os.path.join(holdout_dir, subf), exist_ok=True)

    # 2. Copy .wav files from dir1 and dir2 into combined_dir
    for subf in subfolders:
        src1_files = glob.glob(os.path.join(dir1, subf, "*.wav"))
        src2_files = glob.glob(os.path.join(dir2, subf, "*.wav"))

        for wav_file in src1_files + src2_files:
            shutil.copy2(wav_file, os.path.join(combined_dir, subf))

    # 3. Pick 20% of mixture files for holdout
    mixture_path = os.path.join(combined_dir, "mixture")
    holdout_mixture = os.path.join(holdout_dir, "mixture")
    all_mixture_files = glob.glob(os.path.join(mixture_path, "*.wav"))
    n_mixture = len(all_mixture_files)

    if n_mixture == 0:
        print("No mixture files found. Exiting.")
        return

    n_move = int(n_mixture * fraction)
    # Randomly pick the mixture files to move
    chosen_mixture_files = random.sample(all_mixture_files, n_move)

    # 4. For each chosen mixture file, move it AND any corresponding files
    #    from source1/source2 (if they exist)
    for chosen_file in chosen_mixture_files:
        # Example: chosen_file = "combined/mixture/file123.wav"
        filename = os.path.basename(chosen_file)  # "file123.wav"

        # Move the mixture file itself
        shutil.move(chosen_file, os.path.join(holdout_mixture, filename))

        # Move matching files in source1, source2
        for subf in ("source1", "source2"):
            subfile_path = os.path.join(combined_dir, subf, filename)
            # If file with same name exists, move it
            if os.path.exists(subfile_path):
                shutil.move(subfile_path, os.path.join(holdout_dir, subf, filename))


if __name__ == "__main__":
    # load config
    config_path = os.getenv(
        "CONFIG_PATH",
        "C:\\Users\\arifr\\git\\CommonVoice_RIR\\src\\configs\\combine_and_split_config.json",
    )  # Fallback to a default
    with open(config_path, "r") as f:
        config = json.load(f)
    # fmt: off
    dir1 = config["dir1"]
    dir2 = config["dir2"]
    combined_dir = config["combined"]
    holdout_dir  = config["holdout"]
    fraction = config["fraction"]
    combine_and_split(dir1, dir2, combined_dir, holdout_dir, fraction=0.2)
