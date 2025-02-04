import os
import argparse
import librosa
import multiprocessing
from tqdm import tqdm

def format_duration(seconds):
    """
    Converts seconds into hours, minutes, and remaining seconds.

    Args:
        seconds (float): Total duration in seconds.

    Returns:
        str: Formatted duration as "HH hours, MM minutes, SS seconds".
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours} hours, {minutes} minutes, {seconds:.2f} seconds"

def process_wav_file(file_path):
    """
    Loads a `.wav` file and returns its duration.

    Args:
        file_path (str): Path to the `.wav` file.

    Returns:
        float: Duration of the `.wav` file in seconds.
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)  # Load audio file
        return librosa.get_duration(y=audio, sr=sr)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0.0

def get_total_wav_duration(directory, num_workers=4):
    """
    Calculates the total duration of all `.wav` files in the given directory using multiprocessing.

    Args:
        directory (str): Path to the directory containing `.wav` files.
        num_workers (int): Number of parallel workers for multiprocessing.

    Returns:
        float: Total duration of all `.wav` files in seconds.
    """
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return 0.0

    wav_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]

    if not wav_files:
        print("No .wav files found in the directory.")
        return 0.0

    print(f"Processing {len(wav_files)} .wav files with {num_workers} workers...")

    # Use multiprocessing to process files in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        durations = list(tqdm(pool.imap(process_wav_file, wav_files), total=len(wav_files), desc="Processing WAV files", unit="file"))

    total_duration = sum(durations)
    formatted_duration = format_duration(total_duration)
    print(f"\nTotal duration of all .wav files: {formatted_duration}")

    return total_duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate total duration of .wav files in a directory using multiprocessing.")
    parser.add_argument("--directory", type=str, required=True, help="Path to the directory containing .wav files.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4).")

    args = parser.parse_args()
    get_total_wav_duration(args.directory, args.workers)
