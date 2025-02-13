#!/usr/bin/env python3

"""
Split LDC2014S02 Dataset into Train/Validation/Test by Speaker,
and for each speaker's 2.Silent_Room subfolders, note whether
the following files exist:
  - Mic_CreativeSB.flac
  - Mobile_CreativeSB.flac
  - Yamaha_Mixer.flac
  - OtherFlac (any .flac file *not* one of the three above)

Example Directory Hierarchy:

/export/corpora5/LDC/LDC2014S02/data/
├── Males
│   ├── Session_1
│   │   ├── NS1
│   │   │   ├── 1.Office
│   │   │   ├── 2.Silent_Room  <-- we collect subfolders from here
│   │   │   │   ├── 10.Questions_2
│   │   │   │   │   ├── Mic_CreativeSB.flac
│   │   │   │   │   ├── Mobile_CreativeSB.flac
│   │   │   │   │   ├── Yamaha_Mixer.flac
│   │   │   │   │   └── MyUniqueFile.flac <-- an "other" .flac file
│   │   │   │   ├── ...
│   │   │   └── 3.Cafeteria
│   │   └── ...
│   └── Session_2
│       └── ...
└── Females
    ├── ...
"""

import argparse
import csv
import os
import random
from tqdm import tqdm


def split_dataset(args):
    """
    Perform the speaker-based split of the LDC2014S02 dataset
    and write the resulting CSV file with columns:
      - gender
      - speaker_id
      - subset
      - silent_room_path
      - basename (the subfolder name)
      - Mic_CreativeSB (bool)
      - Mobile_CreativeSB (bool)
      - Yamaha_Mixer (bool)
      - OtherFlac (bool)  <-- any .flac besides the three above
    """

    # Set random seed for reproducibility.
    random.seed(args.seed)

    # We expect two subfolders in base_dir: Males and Females.
    GENDER_DIRS = ["Males", "Females"]

    # speaker_dict will map: (gender, speaker_id) -> list_of_silent_room_paths
    speaker_dict = {}

    # 1) Traverse the dataset structure
    for gender in GENDER_DIRS:
        gender_path = os.path.join(args.base_dir, gender)
        if not os.path.isdir(gender_path):
            print(f"Warning: Directory not found for {gender_path}. Skipping.")
            continue

        # Each gender folder: Session_1, Session_2, ...
        for session_dir in tqdm(os.listdir(gender_path), desc="Processing sessions"):
            session_path = os.path.join(gender_path, session_dir)
            if not os.path.isdir(session_path):
                continue

            # Inside each session, the subdirectories are speaker IDs
            for speaker_id in os.listdir(session_path):
                speaker_path = os.path.join(session_path, speaker_id)
                if not os.path.isdir(speaker_path):
                    continue

                # Specifically want the "2.Silent_Room" folder
                silent_room_path = os.path.join(speaker_path, "2.Silent_Room")

                if os.path.isdir(silent_room_path):
                    # Use (gender, speaker_id) as a unique key
                    key = (gender, speaker_id)
                    if key not in speaker_dict:
                        speaker_dict[key] = []
                    speaker_dict[key].append(silent_room_path)
                else:
                    print(
                        f"Warning: '2.Silent_Room' not found for speaker '{speaker_id}'"
                        f" in session '{session_dir}' under '{gender}'."
                    )

    # 2) Create a list of all unique speakers for splitting
    all_speakers = list(speaker_dict.keys())
    random.shuffle(all_speakers)

    num_speakers = len(all_speakers)
    train_cutoff = int(args.train_ratio * num_speakers)
    val_cutoff = int((args.train_ratio + args.val_ratio) * num_speakers)

    train_speakers = all_speakers[:train_cutoff]
    val_speakers = all_speakers[train_cutoff:val_cutoff]
    test_speakers = all_speakers[val_cutoff:]

    # 3) Write results to CSV
    #
    # We'll add a column "OtherFlac" to indicate if there's any .flac
    # file *other* than the known 3 (Mic_CreativeSB, Mobile_CreativeSB, Yamaha_Mixer).

    known_flac_files = {
        "Mic_CreativeSB.flac",
        "Mobile_CreativeSB.flac",
        "Yamaha_Mixer.flac",
        "Computer_Mic_Front.flac",
    }

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "gender",
                "speaker_id",
                "subset",
                "silent_room_path",
                "basename",
                "Mic_CreativeSB",
                "Mobile_CreativeSB",
                "Yamaha_Mixer",
                "Computer_Mic_Front",
                "OtherFlac",
            ]
        )

        def write_rows_for_subset(speakers, subset_name):
            """Utility to write CSV rows for a given subset."""
            for gender, spk_id in speakers:
                # The speaker may have multiple sessions' 2.Silent_Room paths
                for silent_path in speaker_dict[(gender, spk_id)]:
                    # For each subfolder in 2.Silent_Room:
                    subdirs = [
                        d
                        for d in os.listdir(silent_path)
                        if os.path.isdir(os.path.join(silent_path, d))
                    ]
                    for subfolder in subdirs:
                        subfolder_path = os.path.join(silent_path, subfolder)

                        # Collect all .flac files in this subfolder
                        flac_files = [
                            fname
                            for fname in os.listdir(subfolder_path)
                            if fname.lower().endswith(".flac")
                        ]

                        # Check for the 3 known flacs
                        has_mic = "Mic_CreativeSB.flac" in flac_files
                        has_mobile = "Mobile_CreativeSB.flac" in flac_files
                        has_yamaha = "Yamaha_Mixer.flac" in flac_files
                        has_computer = "Computer_Mic_Front.flac" in flac_files
                        # Check if there's any other .flac besides the known ones
                        other_flacs = set(flac_files) - known_flac_files
                        has_other = len(other_flacs) > 0

                        # Write one row per subfolder
                        writer.writerow(
                            [
                                gender,
                                spk_id,
                                subset_name,
                                subfolder_path,
                                subfolder,  # 'basename' of the subfolder
                                has_mic,
                                has_mobile,
                                has_yamaha,
                                has_computer,
                                has_other,
                            ]
                        )

        # Write for train, val, test
        write_rows_for_subset(train_speakers, "train")
        write_rows_for_subset(val_speakers, "val")
        write_rows_for_subset(test_speakers, "test")

    print(f"Done! Wrote CSV to '{args.output_csv}'.")
    print(
        f"Total speakers: {num_speakers} "
        f"(Train: {len(train_speakers)}, Val: {len(val_speakers)}, Test: {len(test_speakers)})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Split LDC2014S02 dataset by speaker into train/val/test, "
        "and annotate whether each subfolder in 2.Silent_Room "
        "contains known .flac files (Mic_CreativeSB, Mobile_CreativeSB, "
        "Yamaha_Mixer) and/or any *other* .flac file."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/export/corpora5/LDC/LDC2014S02/data",
        help="Base directory of the LDC2014S02 dataset.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/home/afrumme1/CommonVoice_RIR/output_dir/LDC_V2_dataset_creation/ldc_splits.csv",
        help="Output CSV file for splits.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of speakers to assign to the training set.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of speakers to assign to the validation set.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of speakers to assign to the test set.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    split_dataset(args)


if __name__ == "__main__":
    main()
