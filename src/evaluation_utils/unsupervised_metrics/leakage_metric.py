import argparse
import csv
import json
import os
from collections import defaultdict
from typing import List, Tuple

import librosa
import numpy as np
from tqdm import tqdm


def process_arg_list(arg: List[str]) -> List[str]:
    """
    Processes an argument list which may contain comma-separated items.
    For example, if the user provides: --prefixes "intro_,start_" "pre_"
    this function will return: ['intro_', 'start_', 'pre_'].
    """
    result = []
    for item in arg:
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

    base = os.path.splitext(os.path.basename(filepath))[0]
    for prefix in prefixes:
        if base.startswith(prefix):
            base = base[len(prefix) :]
            break
    for suffix in suffixes:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def compute_windowed_correlation_metric(
    file_paths: List[str],
    window_size: int,
    window_step: int,
    top_z: int,
    matrix_output_dir: str,
    top_output_dir: str,
    unique_id: str,
    save_full_matrix: bool = False,
) -> Tuple[float, List[Tuple[float, float, float, float, float]]]:
    """
    Computes a windowed correlation metric. For the two audio files:
      - It extracts sliding windows (of given window_size and window_step).
      - It computes a matrix where each element [i, j] is the Pearson correlation between
        the i-th window of the first audio and the j-th window of the second audio.
      - If save_full_matrix is True, the full correlation matrix is saved as:
            {matrix_output_dir}/windowed_correlation_matrix_{unique_id}.csv
      - The top Z correlation values (and their corresponding start and end times in seconds)
        are identified and saved as:
            {top_output_dir}/top_{top_z}_windowed_correlations_{unique_id}.csv
      - The function returns a tuple:
            (maximum_correlation, top_entries)
        where top_entries is a list of tuples of the form:
            (audio1_start_sec, audio1_end_sec, audio2_start_sec, audio2_end_sec, correlation)
    """
    if len(file_paths) != 2:
        print(
            f"Warning: windowed_correlation metric expects exactly 2 audio signals; got {len(file_paths)}."
        )
        return 0.0, []

    # Load audios.
    data1, sr1 = librosa.load(file_paths[0], sr=None, mono=False)
    data2, sr2 = librosa.load(file_paths[1], sr=None, mono=False)

    if sr1 != sr2:
        raise ValueError(
            f"Sampling rates do not match for files:\n"
            f"{file_paths[0]} (sr={sr1})\n"
            f"{file_paths[1]} (sr={sr2})"
        )

    # Use the first channel if multi-channel.
    if data1.ndim > 1:
        data1 = data1[0]
    if data2.ndim > 1:
        data2 = data2[0]

    if len(data1) != len(data2):
        raise ValueError(
            f"Audio lengths do not match for files:\n"
            f"{file_paths[0]} has length {len(data1)}\n"
            f"{file_paths[1]} has length {len(data2)}"
        )

    n_samples = len(data1)
    # Compute window indices.
    win_indices = list(range(0, n_samples - window_size + 1, window_step))
    n_win = len(win_indices)
    corr_matrix = np.zeros((n_win, n_win))

    # Compute correlation for each pair of windows.
    for i, start1 in enumerate(win_indices):
        window1 = data1[start1 : start1 + window_size]
        for j, start2 in enumerate(win_indices):
            window2 = data2[start2 : start2 + window_size]
            if np.std(window1) == 0 or np.std(window2) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(window1, window2)[0, 1]
            corr_matrix[i, j] = corr

    # Optionally save the full correlation matrix.
    if save_full_matrix:
        matrix_csv = os.path.join(
            matrix_output_dir, f"windowed_correlation_matrix_{unique_id}.csv"
        )
        np.savetxt(matrix_csv, corr_matrix, delimiter=",", fmt="%.6f")
        print(f"Saved full correlation matrix to {matrix_csv}")

    # Use the sampling rate (sr1) for conversion.
    sr = sr1
    flat_indices = np.argsort(corr_matrix, axis=None)[::-1]  # descending order
    top_entries = []
    count = 0
    for flat_idx in flat_indices:
        if count >= top_z:
            break
        i, j = np.unravel_index(flat_idx, corr_matrix.shape)
        corr_val = corr_matrix[i, j]
        audio1_start = win_indices[i]
        audio1_end = audio1_start + window_size
        audio2_start = win_indices[j]
        audio2_end = audio2_start + window_size
        # Convert sample indices to seconds.
        audio1_start_sec = audio1_start / sr
        audio1_end_sec = audio1_end / sr
        audio2_start_sec = audio2_start / sr
        audio2_end_sec = audio2_end / sr
        top_entries.append(
            (
                audio1_start_sec,
                audio1_end_sec,
                audio2_start_sec,
                audio2_end_sec,
                corr_val,
            )
        )
        count += 1

    top_csv = os.path.join(
        top_output_dir, f"top_{top_z}_windowed_correlations_{unique_id}.csv"
    )
    with open(top_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "audio1_start_sec",
                "audio1_end_sec",
                "audio2_start_sec",
                "audio2_end_sec",
                "correlation",
            ]
        )
        writer.writerows(top_entries)
    print(f"Saved top {top_z} windowed correlations to {top_csv}")

    max_corr = np.max(corr_matrix)
    return max_corr, top_entries


def compute_metric(file_paths: List[str], metric: str = "unseparation", **kwargs):
    """
    Dispatches the metric computation. Currently supports:
      - 'leakege' for windowed correlation (detailed output is saved per unique_id).
    For the windowed metric, additional keyword arguments are expected:
        window_size, window_step, top_z, matrix_output_dir, top_output_dir, unique_id, save_full_matrix.
    If 'leakege' is used, the function returns a tuple:
        (max_correlation, top_entries)
    """
    if metric == "leakege":
        return compute_windowed_correlation_metric(file_paths, **kwargs)
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
    prefixes: List[str],
    suffixes: List[str],
    wav_ends_with: str,
    top_x: int = 10,
    metric: str = "unseparation",
    window_size: int = None,
    window_step: int = None,
    top_z: int = None,
    output_summary_dir: str = None,
    output_detail_dir: str = None,
    save_full_matrix: bool = False,
):
    # Use provided output directories or default to audio_dir.
    if output_summary_dir is None:
        output_summary_dir = audio_dir
    if output_detail_dir is None:
        output_detail_dir = audio_dir

    # Create subdirectories for full matrices and top results.
    matrix_output_dir = os.path.join(output_detail_dir, "full_matrix")
    top_output_dir = os.path.join(output_detail_dir, "top_results")
    os.makedirs(matrix_output_dir, exist_ok=True)
    os.makedirs(top_output_dir, exist_ok=True)

    # 1. List all .wav files in the audio directory that end with wav_ends_with.
    wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(wav_ends_with)]

    # 2. Group files by unique ID using the provided prefixes and suffixes.
    id_to_files = defaultdict(list)
    for filename in wav_files:
        full_path = os.path.join(audio_dir, filename)
        unique_id = extract_unique_id(full_path, prefixes, suffixes)
        id_to_files[unique_id].append(full_path)

    # 3. Compute the metric for each unique ID with a progress bar.
    # We store for each unique_id a tuple: (score, top_entries)
    id_to_score = {}
    for unique_id, files in tqdm(id_to_files.items(), desc="Processing unique IDs"):
        try:
            if metric == "leakege":
                max_corr, top_entries = compute_metric(
                    files,
                    metric,
                    window_size=window_size,
                    window_step=window_step,
                    top_z=top_z,
                    matrix_output_dir=matrix_output_dir,
                    top_output_dir=top_output_dir,
                    unique_id=unique_id,
                    save_full_matrix=save_full_matrix,
                )
                score = max_corr  # Use the maximum correlation as the summary metric.
            else:
                score = compute_metric(files, metric)
                top_entries = None
        except Exception as e:
            print(f"Error computing metric for id {unique_id}: {e}")
            score = 0.0
            top_entries = None
        id_to_score[unique_id] = (score, top_entries)

    # 4. Sort the results by score (ascending order) using the maximum correlation.
    sorted_scores = sorted(id_to_score.items(), key=lambda x: x[1][0])

    # Prepare summary data with columns:
    # id, score, audio1_start_sec, audio1_end_sec, audio2_start_sec, audio2_end_sec, top_entries (JSON string)
    summary_data = []
    for uid, (score, top_entries) in sorted_scores:
        if top_entries is not None and len(top_entries) > 0:
            t1, t2, t3, t4, _ = top_entries[0]
        else:
            t1, t2, t3, t4 = "", "", "", ""
        summary_data.append(
            (
                uid,
                score,
                t1,
                t2,
                t3,
                t4,
                json.dumps(top_entries) if top_entries is not None else "",
            )
        )

    # 5. Save all scores to a summary CSV file.
    all_scores_csv = os.path.join(output_summary_dir, "all_scores.csv")
    save_to_csv(
        summary_data,
        [
            "id",
            "score",
            "audio1_start_sec",
            "audio1_end_sec",
            "audio2_start_sec",
            "audio2_end_sec",
            "top_entries",
        ],
        all_scores_csv,
    )

    # 6. Save top X scores (highest values) to a summary CSV file.
    top_scores_data = (
        sorted_scores[-top_x:] if len(sorted_scores) >= top_x else sorted_scores
    )
    top_scores_data = sorted(top_scores_data, key=lambda x: x[1][0], reverse=True)
    top_scores_data = [
        (
            uid,
            score,
            (top_entries[0][0] if top_entries and len(top_entries) > 0 else ""),
            (top_entries[0][1] if top_entries and len(top_entries) > 0 else ""),
            (top_entries[0][2] if top_entries and len(top_entries) > 0 else ""),
            (top_entries[0][3] if top_entries and len(top_entries) > 0 else ""),
            json.dumps(top_entries) if top_entries is not None else "",
        )
        for uid, (score, top_entries) in top_scores_data
    ]
    top_scores_csv = os.path.join(output_summary_dir, f"top_{top_x}_scores.csv")
    save_to_csv(
        top_scores_data,
        [
            "id",
            "score",
            "audio1_start_sec",
            "audio1_end_sec",
            "audio2_start_sec",
            "audio2_end_sec",
            "top_entries",
        ],
        top_scores_csv,
    )

    # 7. Save bottom X scores (lowest values) to a summary CSV file.
    bottom_scores_data = (
        sorted_scores[:top_x] if len(sorted_scores) >= top_x else sorted_scores
    )
    bottom_scores_data = [
        (
            uid,
            score,
            (top_entries[0][0] if top_entries and len(top_entries) > 0 else ""),
            (top_entries[0][1] if top_entries and len(top_entries) > 0 else ""),
            (top_entries[0][2] if top_entries and len(top_entries) > 0 else ""),
            (top_entries[0][3] if top_entries and len(top_entries) > 0 else ""),
            json.dumps(top_entries) if top_entries is not None else "",
        )
        for uid, (score, top_entries) in bottom_scores_data
    ]
    bottom_scores_csv = os.path.join(output_summary_dir, f"bottom_{top_x}_scores.csv")
    save_to_csv(
        bottom_scores_data,
        [
            "id",
            "score",
            "audio1_start_sec",
            "audio1_end_sec",
            "audio2_start_sec",
            "audio2_end_sec",
            "top_entries",
        ],
        bottom_scores_csv,
    )

    # 8. Identify the median score and its immediate neighbors and save to a summary CSV.
    n = len(sorted_scores)
    if n == 0:
        print("No .wav files found in the directory.")
        return

    if n % 2 == 1:
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

    median_data = []
    for uid, (score, top_entries) in median_neighbors:
        if top_entries is not None and len(top_entries) > 0:
            t1, t2, t3, t4, _ = top_entries[0]
        else:
            t1, t2, t3, t4 = "", "", "", ""
        median_data.append(
            (
                uid,
                score,
                t1,
                t2,
                t3,
                t4,
                json.dumps(top_entries) if top_entries is not None else "",
            )
        )
    median_csv = os.path.join(output_summary_dir, "median_and_neighbors.csv")
    save_to_csv(
        median_data,
        [
            "id",
            "score",
            "audio1_start_sec",
            "audio1_end_sec",
            "audio2_start_sec",
            "audio2_end_sec",
            "top_entries",
        ],
        median_csv,
    )


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
        "--prefixes",
        type=str,
        nargs="*",
        default=[""],
        help="List of possible prefixes to remove from the filename. Provide multiple prefixes separated by spaces or as a comma-separated string.",
    )
    parser.add_argument(
        "--suffixes",
        type=str,
        nargs="*",
        default=["_spk1_corrected", "_spk2_corrected"],
        help="List of possible suffixes to remove from the filename. Provide multiple suffixes separated by spaces or as a comma-separated string.",
    )
    parser.add_argument(
        "--top_x",
        type=int,
        default=3,
        help="Number of top and bottom scores to output.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="leakege",
        help="Metric to compute (choose 'unseparation' or 'leakege').",
    )
    # Additional parameters for the windowed (leakege) correlation metric.
    parser.add_argument(
        "--window_size",
        type=int,
        default=8000,
        help="Window size (in samples) for the windowed correlation metric.",
    )
    parser.add_argument(
        "--window_step",
        type=int,
        default=4000,
        help="Window step (in samples) for the windowed correlation metric.",
    )
    parser.add_argument(
        "--top_z",
        type=int,
        default=10,
        help="Number of top correlating windows to output (for windowed correlation).",
    )
    # New arguments for output directories.
    parser.add_argument(
        "--output_summary_dir",
        type=str,
        default="C:\\Users\\arifr\\git\\CommonVoice_RIR\\src\\evaluation_utils\\unsupervised_metrics\\summary_dir",
        help="Directory to save summary CSV files (all_scores, top scores, etc.). Defaults to audio_dir if not provided.",
    )
    parser.add_argument(
        "--output_detail_dir",
        type=str,
        default="C:\\Users\\arifr\\git\\CommonVoice_RIR\\src\\evaluation_utils\\unsupervised_metrics\\detailed_dir",
        help="Directory to save detailed per-unique-ID CSV files (e.g., windowed correlation matrices). Defaults to audio_dir if not provided.",
    )
    parser.add_argument(
        "--wav_ends_with",
        type=str,
        default="_corrected.wav",
        help="Process only files that end with this string.",
    )
    parser.add_argument(
        "--save_full_matrix",
        action="store_true",
        help="If provided, save the full correlation matrix CSV files. Otherwise, skip saving them.",
    )
    args = parser.parse_args()

    prefixes = process_arg_list(args.prefixes)
    suffixes = process_arg_list(args.suffixes)
    output_summary_dir = (
        args.output_summary_dir if args.output_summary_dir else args.audio_dir
    )
    output_detail_dir = (
        args.output_detail_dir if args.output_detail_dir else args.audio_dir
    )

    main(
        args.audio_dir,
        prefixes,
        suffixes,
        wav_ends_with=args.wav_ends_with,
        top_x=args.top_x,
        metric=args.metric,
        window_size=args.window_size,
        window_step=args.window_step,
        top_z=args.top_z,
        output_summary_dir=output_summary_dir,
        output_detail_dir=output_detail_dir,
        save_full_matrix=args.save_full_matrix,
    )
