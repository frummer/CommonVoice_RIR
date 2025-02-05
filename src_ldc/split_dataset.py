import argparse
import os
import random
import shutil

from tqdm import tqdm


def copy_files(
    ids_list: list, sub_dir: str, split_name: str, base_dir: str, output_dir: str
):
    """
    Copies .wav files specified in `ids_list` from:
        base_dir/sub_dir --> output_dir/split_name/sub_dir

    Handles the special 'compressed_mixture' prefix "comp_".

    :param ids_list:   List of file IDs to copy.
    :param sub_dir:    Name of the subdirectory (e.g. mixture, source1, compressed_mixture).
    :param split_name: Which split: train, validation, or test.
    :param base_dir:   Path to the base input dataset.
    :param output_dir: Path to the output dataset.
    """
    # Source folder (where we read the files from):
    src_sub_dir = os.path.join(base_dir, sub_dir)

    # Destination folder (where we will copy them to):
    # e.g. output_dir/train/mixture/
    dst_sub_dir = os.path.join(output_dir, split_name, sub_dir)
    os.makedirs(dst_sub_dir, exist_ok=True)

    for file_id in ids_list:
        # Handle "compressed_mixture" prefix
        if sub_dir == "compressed_mixture":
            filename = f"comp_{file_id}.wav"
        else:
            filename = f"{file_id}.wav"

        src = os.path.join(src_sub_dir, filename)
        dst = os.path.join(dst_sub_dir, filename)

        if os.path.isfile(src):
            # Copy the file (or use shutil.move if desired)
            shutil.copy2(src, dst)
        else:
            print(f"Warning: File not found: {src}")


def split_dataset(
    base_dir: str,
    output_dir: str,
    sub_dirs: list,
    reference_dir: str,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
):
    """
    Split the dataset into train/validation/test given explicit parameters.

    :param base_dir:       Path to the base input dataset directory.
    :param output_dir:     Path where train, validation, and test folders will be created.
    :param sub_dirs:       List of subdirectories containing .wav files.
    :param reference_dir:  The subdirectory that will be used to collect file IDs.
                           If None, defaults to the first in sub_dirs.
    :param train_ratio:    Ratio of total files for the training set.
    :param validation_ratio: Ratio of total files for the validation set.
    :param seed:           Random seed for reproducible splits.
    """

    # Set the random seed for reproducible splits
    random.seed(seed)

    # 1. Determine which subdirectory we'll use as the "reference" for collecting IDs
    if reference_dir:
        if reference_dir not in sub_dirs:
            raise ValueError(
                f"Error: reference_dir '{reference_dir}' is not in the list of sub_dirs {sub_dirs}."
            )
        ref_subdir = reference_dir
    else:
        # If no reference_dir provided, we use the first in sub_dirs
        ref_subdir = sub_dirs[0]

    reference_path = os.path.join(base_dir, ref_subdir)

    # 2. Get all .wav filenames from the reference subdirectory
    all_filenames = sorted(f for f in os.listdir(reference_path) if f.endswith(".wav"))

    # 3. Extract IDs (assuming the IDs are everything before ".wav")
    #    e.g. "abc123.wav" -> "abc123"
    all_ids = [os.path.splitext(fn)[0] for fn in all_filenames]

    # 4. Shuffle the IDs
    random.shuffle(all_ids)

    # 5. Determine how many items in each split
    total_count = len(all_ids)
    train_count = int(train_ratio * total_count)
    validation_count = int(validation_ratio * total_count)
    test_count = total_count - train_count - validation_count  # ensures sum = total

    train_ids = all_ids[:train_count]
    validation_ids = all_ids[train_count : train_count + validation_count]
    test_ids = all_ids[train_count + validation_count :]

    print(f"Reference subdirectory: '{ref_subdir}'")
    print(f"Total .wav files found: {total_count}")
    print(f"Train split:            {len(train_ids)}")
    print(f"Validation split:       {len(validation_ids)}")
    print(f"Test split:             {len(test_ids)}")

    # 6. Create the main output folders: train, validation, test
    split_names = ["train", "validation", "test"]
    for split_name in split_names:
        os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

    # 7. Copy files into train/validation/test for each subdirectory
    for split_name, ids_list in tqdm(
        zip(split_names, [train_ids, validation_ids, test_ids]),
        desc="Splitting into train/validation/test",
        total=len(split_names),
    ):
        for sd in tqdm(
            sub_dirs, desc=f"Copying for {split_name}", leave=False, total=len(sub_dirs)
        ):
            copy_files(ids_list, sd, split_name, base_dir, output_dir)

    print("Done! Files have been split into train/validation/test folders.")
    print(f"Output structure is located in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/validation/test."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to the base input dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path where train, validation, and test folders will be created.",
    )
    parser.add_argument(
        "--sub_dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of subdirectories containing .wav files.",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default=None,
        help="Name of the subdirectory that will be used to collect file IDs. "
        "If not provided, defaults to the first in --sub_dirs.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of total files for the training set (default: 0.8).",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
        help="Ratio of total files for the validation set (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )

    args = parser.parse_args()

    # Pass parameters individually (not the entire args) to split_dataset
    split_dataset(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        sub_dirs=args.sub_dirs,
        reference_dir=args.reference_dir,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
