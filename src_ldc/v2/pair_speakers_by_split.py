#!/usr/bin/env python3

"""
Create cross-speaker utterance pairs from an LDC splits CSV, 
one CSV per subset (train/val/test), in a SINGLE DIRECTION:
  - We only create (A, B), not (B, A).

We only pair utterances:
  - From the same microphone
  - In the same subset
  - From different unique speakers

Output: three files, placed in --output_dir:
   train_pairs.csv
   val_pairs.csv
   test_pairs.csv

Columns in each output file:
   mic, unique_speaker_id_1, path_1, unique_speaker_id_2, path_2
"""

import argparse
import csv
import os


def create_pairs(args):
    """
    Reads the input CSV (with columns like 'gender', 'speaker_id',
    'subset', 'basename', 'Mic_CreativeSB', 'Mobile_CreativeSB',
    'Yamaha_Mixer', etc.). For each microphone that is marked True,
    we build a (subset -> mic -> list_of_utterances) structure.

    Then, for each subset, we create a CSV named '{subset}_pairs.csv'
    in the output directory, containing all cross-speaker pairs for
    each microphone, but ONLY in one direction (i < j).
    """
    # Microphones of interest:
    mic_list = [
        "Mic_CreativeSB",
        "Mobile_CreativeSB",
        "Yamaha_Mixer",
        "Computer_Mic_Front",
    ]
    # subset_data[subset][mic] = list of (unique_speaker_id, full_flac_path)
    subset_data = {
        "train": {mic: [] for mic in mic_list},
        "val": {mic: [] for mic in mic_list},
        "test": {mic: [] for mic in mic_list},
    }

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Read the input CSV
    with open(args.input_csv, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Basic fields
            gender = row["gender"]  # e.g. "Males" or "Females"
            speaker_id = row["speaker_id"]  # e.g. "NS1"
            subset = row["subset"]  # e.g. "train", "val", "test"

            # Build a unique speaker ID
            unique_id = f"{gender}_{speaker_id}"  # e.g. "Males_NS1"

            if subset not in subset_data:
                # If we find a row that doesn't map to train/val/test, skip it
                continue

            # Path info
            silent_path = row["silent_room_path"]
            basename = row["basename"]

            # For each microphone, check if "True"
            for mic in mic_list:
                mic_val = row[mic].strip().lower()  # "true" or "false"
                if mic_val == "true":
                    # Construct the .flac path
                    flac_path = os.path.join(silent_path, f"{mic}.flac")
                    subset_data[subset][mic].append((unique_id, flac_path))

    # 2) For each subset, create an output CSV file
    #    e.g. train_pairs.csv, val_pairs.csv, test_pairs.csv

    for subset in ["train", "val", "test"]:
        outfile = os.path.join(args.output_dir, f"{subset}_pairs.csv")
        with open(outfile, "w", newline="") as out_f:
            writer = csv.writer(out_f)
            # We skip 'subset' column since each file corresponds to a single subset
            writer.writerow(
                [
                    "mic",
                    "unique_speaker_id_1",
                    "path_1",
                    "unique_speaker_id_2",
                    "path_2",
                ]
            )

            # For each microphone within this subset, do cross-speaker pairing
            for mic in mic_list:
                utterances = subset_data[subset][mic]
                n = len(utterances)

                # Single-direction approach:
                # for j in range(i+1, n) => NO (B, A)
                for i in range(n):
                    spk_i, path_i = utterances[i]

                    for j in range(i + 1, n):
                        spk_j, path_j = utterances[j]

                        # Only pair if different unique speaker IDs
                        if spk_i != spk_j:
                            writer.writerow([mic, spk_i, path_i, spk_j, path_j])

        print(f"Created SINGLE-DIRECTION pairs for '{subset}' subset: {outfile}")


def main():
    parser = argparse.ArgumentParser(
        description="Create cross-speaker utterance pairs from the LDC CSV, "
        "generating one CSV per subset (train, val, test). "
        "Pairs are SINGLE-DIRECTION only (i < j). "
        "We only pair utterances with the same microphone and different unique speakers."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/home/afrumme1/CommonVoice_RIR/output_dir/LDC_V2_dataset_creation/ldc_splits.csv",
        help="Input CSV file from the previous script (with microphone columns).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/afrumme1/CommonVoice_RIR/output_dir/LDC_V2_dataset_creation",
        help="Output directory for the {subset}_pairs.csv files.",
    )
    args = parser.parse_args()

    create_pairs(args)


if __name__ == "__main__":
    main()
