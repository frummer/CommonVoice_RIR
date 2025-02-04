import os
import argparse
from collections import defaultdict

def normalize_filename(filename, possible_prefixes):
    """
    Normalize a filename by removing possible prefixes.

    Args:
        filename (str): The original filename.
        possible_prefixes (list): List of possible prefixes to remove.

    Returns:
        str: The normalized filename without the prefix.
    """
    for prefix in possible_prefixes:
        if filename.startswith(prefix):
            return filename[len(prefix):]
    return filename

def check_file_consistency(base_dir, possible_prefixes=None):
    """
    Checks if all subdirectories contain the same named files
    or the same named files with an optional prefix.

    Args:
        base_dir (str): The base directory containing subdirectories.
        possible_prefixes (list, optional): List of possible prefixes to ignore.
    
    Returns:
        None: Prints mismatched files if any.
    """
    if not os.path.isdir(base_dir):
        print(f"Error: '{base_dir}' is not a valid directory.")
        return
    
    possible_prefixes = possible_prefixes or []
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    if not subdirs:
        print("No subdirectories found.")
        return
    
    file_sets = defaultdict(set)

    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        files = os.listdir(subdir_path)
        normalized_files = {normalize_filename(f, possible_prefixes) for f in files}
        file_sets[subdir] = normalized_files

    # Use the first subdirectory as the reference
    reference_subdir = subdirs[0]
    reference_files = file_sets[reference_subdir]

    print(f"Using '{reference_subdir}' as reference.")
    
    mismatches = {}
    
    for subdir, files in file_sets.items():
        if files != reference_files:
            mismatches[subdir] = files.symmetric_difference(reference_files)

    if mismatches:
        print("\nMismatched files detected:")
        for subdir, diff_files in mismatches.items():
            print(f"  {subdir}: {diff_files}")
    else:
        print("All subdirectories contain the same files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if all subdirectories have the same named files.")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to the base directory.")
    parser.add_argument("--prefixes", type=str, nargs="*", default=[], 
                        help="List of possible prefixes to ignore.")

    args = parser.parse_args()
    check_file_consistency(args.base_dir, args.prefixes)
