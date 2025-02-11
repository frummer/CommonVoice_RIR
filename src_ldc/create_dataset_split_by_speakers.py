#!/usr/bin/env python3

"""
Split LDC2014S02 Dataset into Train/Validation/Test by Speaker

Example Directory Hierarchy:

/export/corpora5/LDC/LDC2014S02/data/
├── Females
│   ├── Session_1
│   │   ├── NS1
│   │   │   ├── 1.Office
│   │   │   ├── 2.Silent_Room       <-- we collect paths from here
│   │   │   └── 3.Cafeteria
│   │   └── ...
│   └── Session_2
│       └── ...
├── Males
│   ├── Session_1
│   │   ├── NS1
│   │   │   ├── 1.Office
│   │   │   ├── 2.Silent_Room       <-- we collect paths from here
│   │   │   └── 3.Cafeteria
│   │   └── ...
│   ├── Session_2
│   └── Session_3

We assume:
- Each top-level directory (Males / Females) contains multiple session directories (Session_1, Session_2, Session_3, ...).
- Each session directory contains multiple speaker subfolders (S10, NS11, etc.).
- Each speaker directory contains (optionally) "2.Silent_Room" where the relevant files reside.

This script:
1. Collects each speaker's `2.Silent_Room` path(s).
2. Splits the dataset by speaker into train/val/test (80/10/10).
3. Writes a CSV with columns: [gender, speaker_id, subset, path_to_silent_room].
"""

import os
import csv
import random
import argparse


def split_dataset(args):
    """
    Perform the speaker-based split of the LDC2014S02 dataset
    and write the resulting CSV file.
    """
    # Set random seed for reproducibility.
    random.seed(args.seed)
    
    # We expect two subfolders in base_dir: Males and Females
    GENDER_DIRS = ["Males", "Females"]
    
    # speaker_dict will map: (gender, speaker_id) -> list_of_silent_room_paths
    speaker_dict = {}
    
    # 1) Traverse the dataset structure
    for gender in GENDER_DIRS:
        gender_path = os.path.join(args.base_dir, gender)
        if not os.path.isdir(gender_path):
            print(f"Warning: Directory not found for {gender_path}. Skipping.")
            continue
        
        # Each gender folder has subfolders: Session_1, Session_2, Session_3, ...
        for session_dir in os.listdir(gender_path):
            session_path = os.path.join(gender_path, session_dir)
            if not os.path.isdir(session_path):
                continue
            
            # Inside each session directory, the subdirectories are speaker IDs
            for speaker_id in os.listdir(session_path):
                speaker_path = os.path.join(session_path, speaker_id)
                if not os.path.isdir(speaker_path):
                    continue
                
                # We specifically want the "2.Silent_Room" folder
                silent_room_path = os.path.join(speaker_path, "2.Silent_Room")
                
                if os.path.isdir(silent_room_path):
                    # Use (gender, speaker_id) as a unique key
                    key = (gender, speaker_id)
                    if key not in speaker_dict:
                        speaker_dict[key] = []
                    speaker_dict[key].append(silent_room_path)
                else:
                    print(f"Warning: '2.Silent_Room' not found for speaker '{speaker_id}'"
                          f" in session '{session_dir}' under '{gender}'.")
    
    # 2) Create a list of all unique speakers for splitting
    all_speakers = list(speaker_dict.keys())
    random.shuffle(all_speakers)
    
    num_speakers = len(all_speakers)
    train_cutoff = int(args.train_ratio * num_speakers)
    val_cutoff   = int((args.train_ratio + args.val_ratio) * num_speakers)
    
    train_speakers = all_speakers[:train_cutoff]
    val_speakers   = all_speakers[train_cutoff:val_cutoff]
    test_speakers  = all_speakers[val_cutoff:]
    
    # 3) Write results to CSV
    # Columns: gender, speaker_id, subset, path_to_silent_room
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gender", "speaker_id", "subset", "path_to_silent_room"])
        
        for (gender, spk_id) in train_speakers:
            for p in speaker_dict[(gender, spk_id)]:
                writer.writerow([gender, spk_id, "train", p])
        
        for (gender, spk_id) in val_speakers:
            for p in speaker_dict[(gender, spk_id)]:
                writer.writerow([gender, spk_id, "val", p])
        
        for (gender, spk_id) in test_speakers:
            for p in speaker_dict[(gender, spk_id)]:
                writer.writerow([gender, spk_id, "test", p])
    
    print(f"Done! Wrote CSV to '{args.output_csv}'.")
    print(f"Total speakers: {num_speakers} "
          f"(Train: {len(train_speakers)}, Val: {len(val_speakers)}, Test: {len(test_speakers)})")


def main():
    parser = argparse.ArgumentParser(
        description="Split LDC2014S02 dataset by speaker into train/val/test."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/export/corpora5/LDC/LDC2014S02/data",
        help="Base directory of the LDC2014S02 dataset."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="ldc_splits.csv",
        help="Output CSV file for splits."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of speakers to assign to the training set."
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of speakers to assign to the validation set."
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of speakers to assign to the test set."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility."
    )
    
    args = parser.parse_args()
    split_dataset(args)


if __name__ == "__main__":
    main()
