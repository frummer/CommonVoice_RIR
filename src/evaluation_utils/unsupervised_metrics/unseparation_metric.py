import argparse
import csv
import os
from collections import defaultdict
from typing import List, Tuple
from tqdm import tqdm

import librosa
import numpy as np


def process_arg_list(arg: List[str]) -> List[str]:
    """
    Processes an argument list which may contain comma-separated items.
    For example, if the user provides: --prefixes "intro_,start_" "pre_"
    this function will return: ['intro_', 'start_', 'pre_'].
    """
    result = []
    for item in arg:
        # Split each item by comma and remove extra whitespace.
        result.extend([x.strip() for x in item.split(",") if x.strip()])
    return result


def extract_unique_id(
    filepath: str, prefixes: List[str] = None, suffixes: List[str] = None
) -> str:
    """
    Extract the unique ID from a filename (or full path) by removing any one of the given
    possible prefixes and suffixes from the file’s basename (without its extension).

    For example, if the basename is "intro_12345_outro" and
      prefixes = ["intro_", "start_"]
      suffixes = ["_outro", "_end"]
    then the returned unique ID will be "12345".
    """
    prefixes = prefixes or []
    suffixes = suffixes or []

    # Get the basename without the file extension.
    base = os.path.splitext(os.path.basename(filepath))[0]

    # Remove the first matching prefix (if any).
    for prefix in prefixes:
        if base.startswith(prefix):
            base = base[len(prefix) :]
            break  # Remove only one prefix.

    # Remove the first matching suffix (if any).
    for suffix in suffixes:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break  # Remove only one suffix.

    return base


def compute_unseparation_metric(file_paths: List[str]) -> float:
    """
    Computes the unseparation metric, defined here as the Pearson correlation coefficient
    between two audio signals.

    Expects exactly 2 file paths; if not, prints a warning and returns 0.0.
    Uses librosa to load the audio files with their native sampling rates.
    Raises a ValueError if:
      - The sampling rates do not match.
      - The audio signals are not the same length.
    """
    if len(file_paths) != 2:
        print(
            f"Warning: unseparation metric expects exactly 2 audio signals; got {len(file_paths)}."
        )
        return 0.0

    # Load the two audio files with their native sampling rates (no resampling).
    data1, sr1 = librosa.load(file_paths[0], sr=None, mono=False)
    data2, sr2 = librosa.load(file_paths[1], sr=None, mono=False)

    # Raise error if sampling rates differ.
    if sr1 != sr2:
        raise ValueError(
            f"Sampling rates do not match for files:\n"
            f"{file_paths[0]} (sr={sr1})\n"
            f"{file_paths[1]} (sr={sr2})"
        )

    # If the audio data has more than one channel, select the first channel.
    if data1.ndim > 1:
        data1 = data1[0]
    if data2.ndim > 1:
        data2 = data2[0]

    # Validate that the audio signals are of the same length.
    if len(data1) != len(data2):
        raise ValueError(
            f"Audio lengths do not match for files:\n"
            f"{file_paths[0]} has length {len(data1)}\n"
            f"{file_paths[1]} has length {len(data2)}"
        )

    # Check if either signal is constant.
    if np.std(data1) == 0 or np.std(data2) == 0:
        print(
            "Warning: one of the signals is constant; correlation is undefined. Returning 0.0."
        )
        return 0.0

    # Compute Pearson correlation coefficient.
    corr_matrix = np.corrcoef(data1, data2)
    correlation = corr_matrix[0, 1]
    return correlation


def compute_metric(file_paths: List[str], metric: str = "unseparation") -> float:
    """
    Dispatches the metric computation. Currently, only the 'unseparation' metric is supported.
    """
    if metric == "unseparation":
        return compute_unseparation_metric(file_paths)
    else:
        raise ValueError("Unknown metric: {}".format(metric))


def save_to_csv(
    data: List[Tuple[str, float]], headers: List[str], output_path: str
) -> None:
    """
    Save a list of tuples (or lists) to a CSV file.
    Each tuple should match the order of the headers.
    """
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)


def main(
    audio_dir: str,
    output_dir: str,
    prefixes: List[str],
    suffixes: List[str],
    wav_ends_with: str,
    top_x: int = 10,
    metric: str = "unseparation",
):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. List all .wav files in the audio directory.
    wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(wav_ends_with)]

    # 2. Group files by unique ID using the provided prefixes and suffixes.
    id_to_files = defaultdict(list)
    for filename in wav_files:
        full_path = os.path.join(audio_dir, filename)
        unique_id = extract_unique_id(full_path, prefixes, suffixes)
        id_to_files[unique_id].append(full_path)

    # 3. Compute the metric for each unique ID.
    id_to_score = {}
    for unique_id, files in tqdm(id_to_files.items(), desc="Computing metrics", unit="file"):
        try:
            score = compute_metric(files, metric)
        except Exception as e:
            print(f"Error computing metric for id {unique_id}: {e}")
            score = 0.0
        id_to_score[unique_id] = score

    # 4. Sort the results by score (ascending order).
    sorted_scores = sorted(id_to_score.items(), key=lambda x: x[1])

    # 5. Save all scores to a CSV file.
    all_scores_csv = os.path.join(output_dir, "all_scores.csv")
    save_to_csv(sorted_scores, ["id", "score"], all_scores_csv)

    # 6. Save top X scores (highest values) to a CSV.
    top_scores = (
        sorted_scores[-top_x:] if len(sorted_scores) >= top_x else sorted_scores
    )
    top_scores = sorted(top_scores, key=lambda x: x[1], reverse=True)
    top_scores_csv = os.path.join(output_dir, f"top_{top_x}_scores.csv")
    save_to_csv(top_scores, ["id", "score"], top_scores_csv)

    # 7. Save bottom X scores (lowest values) to a CSV.
    bottom_scores = (
        sorted_scores[:top_x] if len(sorted_scores) >= top_x else sorted_scores
    )
    bottom_scores_csv = os.path.join(output_dir, f"bottom_{top_x}_scores.csv")
    save_to_csv(bottom_scores, ["id", "score"], bottom_scores_csv)

    # 8. Identify the median score and its immediate neighbors.
    n = len(sorted_scores)
    if n == 0:
        print("No .wav files found in the directory.")
        return

    if n % 2 == 1:
        # Odd number of items: one median.
        median_index = n // 2
        median_item = sorted_scores[median_index]
        neighbors = []
        if median_index - 1 >= 0:
            neighbors.append(sorted_scores[median_index - 1])
        neighbors.append(median_item)
        if median_index + 1 < n:
            neighbors.append(sorted_scores[median_index + 1])
        median_neighbors = neighbors
    else:
        # Even number of items: two medians.
        median_index = (n // 2) - 1
        median_item1 = sorted_scores[median_index]
        median_item2 = sorted_scores[median_index + 1]
        neighbors = []
        if median_index - 1 >= 0:
            neighbors.append(sorted_scores[median_index - 1])
        neighbors.extend([median_item1, median_item2])
        if median_index + 2 < n:
            neighbors.append(sorted_scores[median_index + 2])
        median_neighbors = neighbors

    median_csv = os.path.join(output_dir, "median_and_neighbors.csv")
    save_to_csv(median_neighbors, ["id", "score"], median_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute audio metrics for WAV files in a directory."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="C:\\Users\\arifr\\git\\CommonVoice_RIR\\src\\evaluation_utils\\separation_output_prev",
        help="Path to the directory containing WAV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:\\Users\\arifr\\git\\CommonVoice_RIR\\src\\evaluation_utils\\separation_output_prev",
        help="Path to the directory where CSV output files will be saved.",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        nargs="*",
        default=[""],
        help=(
            "List of possible prefixes to remove from the filename. "
            "Provide multiple prefixes separated by spaces or as a comma-separated string."
        ),
    )
    parser.add_argument(
        "--suffixes",
        type=str,
        nargs="*",
        default=["_spk1_corrected", "_spk2_corrected"],
        help=(
            "List of possible suffixes to remove from the filename. "
            "Provide multiple suffixes separated by spaces or as a comma-separated string."
        ),
    )
    parser.add_argument(
        "--top_x",
        type=int,
        default=10,
        help="Number of top and bottom scores to output.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="unseparation",
        help="Metric to compute (default: unseparation).",
    )
    parser.add_argument(
        "--wav_ends_with",
        type=str,
        default="_corrected.wav",
        help="Metric to compute (default: unseparation).",
    )

    args = parser.parse_args()

    # Process prefixes and suffixes in case they are provided as comma-separated strings.
    prefixes = process_arg_list(args.prefixes)
    suffixes = process_arg_list(args.suffixes)

    main(
        args.audio_dir,
        args.output_dir,
        prefixes=prefixes,
        suffixes=suffixes,
        wav_ends_with=args.wav_ends_with,
        top_x=args.top_x,
        metric=args.metric,
    )
