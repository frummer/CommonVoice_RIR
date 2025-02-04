import os
import argparse
from tqdm import tqdm

def count_wav_files_in_subdirs(base_dir):
    """
    Counts the number of `.wav` files in each subdirectory of the given base directory.

    Args:
        base_dir (str): The path to the base directory containing subdirectories.

    Returns:
        None: Prints the number of `.wav` files in each subdirectory.
    
    Notes:
        - If the provided `base_dir` is not a valid directory, an error message is printed.
        - Uses `tqdm` to display a progress bar while iterating through subdirectories.
    """
    if not os.path.isdir(base_dir):
        print(f"Error: '{base_dir}' is not a valid directory.")
        return
    
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    wav_counts = {}
    for subdir in tqdm(subdirs, desc="Processing subdirectories", unit="dir"):
        subdir_path = os.path.join(base_dir, subdir)
        wav_files = [f for f in os.listdir(subdir_path) if f.endswith('.wav')]
        wav_counts[subdir] = len(wav_files)
    
    for subdir, count in wav_counts.items():
        print(f"{subdir}: {count} .wav files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count .wav files in each subdirectory.")
    parser.add_argument("--base_dir", type=str, help="Path to the base directory.")

    args = parser.parse_args()
    count_wav_files_in_subdirs(args.base_dir)
