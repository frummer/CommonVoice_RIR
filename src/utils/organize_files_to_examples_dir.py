import argparse
import os
import shutil
from collections import defaultdict


def organize_files(base_dir, source_dirs, input_dirs, separated_dirs):
    """
    Organizes WAV files into example_{uuid} folders based on the portion of the filename before the first underscore.

    Args:
        base_dir (str): The base directory containing the folders to scan.
        source_dirs (list): List of source directories containing original files.
        input_dirs (list): List of input directories containing input files.
        separated_dirs (list): List of directories containing separated output files.
    """
    # This dictionary will hold entries like:
    # {
    #   'uuid': {
    #       'source': [(file_path, folder_name), ...],
    #       'input': [(file_path, folder_name), ...],
    #       'separated': [(file_path, folder_name), ...]
    #   },
    #   ...
    # }
    files_by_uuid = defaultdict(lambda: {"source": [], "input": [], "separated": []})

    def get_uuid(filename: str) -> str:
        # Strip extension
        base_name = os.path.splitext(filename)[0]
        # For example, a file named:
        #   0a0270bc-1ad2-4201-8572-79b26bfd4c34_spk2.wav
        # becomes:
        #   0a0270bc-1ad2-4201-8572-79b26bfd4c34 (after removing .wav)
        # then we split by underscore:
        #   ["0a0270bc-1ad2-4201-8572-79b26bfd4c34", "spk2"]
        # We keep the first part.
        return base_name.split("_")[0]

    # Collect files from source directories
    for folder in source_dirs:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(".wav"):
                uuid = get_uuid(filename)
                files_by_uuid[uuid]["source"].append((file_path, folder))

    # Collect files from input directories
    for folder in input_dirs:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(".wav"):
                uuid = get_uuid(filename)
                files_by_uuid[uuid]["input"].append((file_path, folder))

    # Collect files from separated output directories
    for folder in separated_dirs:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(".wav"):
                uuid = get_uuid(filename)
                files_by_uuid[uuid]["separated"].append((file_path, folder))

    for uuid, categories in files_by_uuid.items():
        example_folder = os.path.join(base_dir, f"example_{uuid}")
        os.makedirs(example_folder, exist_ok=True)

        # Copy and rename source files
        for file_path, folder in categories["source"]:
            filename = os.path.basename(file_path)
            # Force s1/s2 prefix to differentiate
            if folder.lower() == "source1":
                prefix = "s1"
            elif folder.lower() == "source2":
                prefix = "s2"
            else:
                prefix = folder

            new_filename = f"{prefix}_{filename}"
            new_file_path = os.path.join(example_folder, new_filename)
            shutil.copy(file_path, new_file_path)
            print(f"Copied {file_path} -> {new_file_path}")

        # Copy input and separated files without renaming
        for file_path, folder in categories["input"] + categories["separated"]:
            filename = os.path.basename(file_path)
            new_file_path = os.path.join(example_folder, filename)
            shutil.copy(file_path, new_file_path)
            print(f"Copied {file_path} -> {new_file_path}")

    print("File organization completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Organize WAV files by UUID and rename as needed."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="C:\\Users\\arifr\\Babylon\\separation_analysis",
        help="Base directory containing the folders.",
    )
    parser.add_argument(
        "--source_dirs",
        nargs="*",
        default=["source1", "source2"],
        help="List of source directories.",
    )
    parser.add_argument(
        "--input_dirs",
        nargs="*",
        default=["separation_input"],
        help="List of input directories.",
    )
    parser.add_argument(
        "--separated_dirs",
        nargs="*",
        default=["separation_output"],
        help="List of separated output directories.",
    )

    args = parser.parse_args()
    organize_files(
        base_dir=args.base_dir,
        source_dirs=args.source_dirs,
        input_dirs=args.input_dirs,
        separated_dirs=args.separated_dirs,
    )


if __name__ == "__main__":
    main()
