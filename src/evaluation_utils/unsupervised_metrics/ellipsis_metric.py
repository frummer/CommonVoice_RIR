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
    result = []
    for item in arg:
        result.extend([x.strip() for x in item.split(",") if x.strip()])
    return result


def extract_unique_id(
    filepath: str, prefixes: List[str] = None, suffixes: List[str] = None
) -> str:
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


def compute_ellipsis_metric(file_paths: List[str]) -> Tuple[float, np.ndarray]:
    """
    Computes the ellipsis metric given three file paths:
      [source1, source2, mixture].
    It solves the least-squares problem S w = m, where S is built from the two separated
    signals, and m is the original mixture. It returns:
       mse : Mean Squared Error between m and the reconstructed mixture.
       w   : The coefficient vector [a, b]^T.
    """
    if len(file_paths) != 3:
        raise ValueError(
            "Ellipsis metric expects exactly 3 files (2 separated tracks and 1 mixture)."
        )

    s1, sr1 = librosa.load(file_paths[0], sr=None, mono=False)
    s2, sr2 = librosa.load(file_paths[1], sr=None, mono=False)
    m, sr_m = librosa.load(file_paths[2], sr=None, mono=False)

    if sr1 != sr2 or sr1 != sr_m:
        raise ValueError("Sampling rates do not match among the files.")

    min_len = min(len(s1), len(s2), len(m))
    s1, s2, m = s1[:min_len], s2[:min_len], m[:min_len]

    S = np.column_stack((s1, s2))
    w, residuals, rank, s_vals = np.linalg.lstsq(S, m, rcond=None)
    if rank < 2:
        print("rank<2", flush=True)
        return 1, [-1, -1]
    m_hat = S @ w
    mse = np.mean((m - m_hat) ** 2)
    return mse, w


def save_to_csv(data: List[Tuple], headers: List[str], output_path: str) -> None:
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)


def main(
    separated_audio_dir: str,
    mixtures_audio_dir: str,
    prefixes: List[str],
    suffixes: List[str],
    wav_ends_with: str,
    top_x: int = 10,
    output_dir: str = None,
):
    # Use output_dir for CSV files; if not provided, default to separated_audio_dir.
    if output_dir is None:
        output_dir = separated_audio_dir

    # 1. List all separated audio files in separated_audio_dir that end with wav_ends_with.
    sep_files = [
        f for f in os.listdir(separated_audio_dir) if f.lower().endswith(wav_ends_with)
    ]

    # 2. Group separated files by unique ID.
    id_to_files = defaultdict(list)
    for filename in sep_files:
        full_path = os.path.join(separated_audio_dir, filename)
        uid = extract_unique_id(full_path, prefixes, suffixes)
        id_to_files[uid].append(full_path)

    # 3. For each unique ID, get the two separated files and the corresponding mixture file.
    id_to_metric = {}
    for uid, sep_list in tqdm(id_to_files.items(), desc="Processing unique IDs"):
        if len(sep_list) != 2:
            print(
                f"Warning: For id {uid}, expected 2 separated files, but got {len(sep_list)}. Skipping."
            )
            continue
        mix_path = os.path.join(mixtures_audio_dir, uid + ".wav")
        if not os.path.exists(mix_path):
            print(
                f"Warning: Mixture file for id {uid} not found at {mix_path}. Skipping."
            )
            continue
        file_paths = [sep_list[0], sep_list[1], mix_path]
        try:
            mse, w = compute_ellipsis_metric(file_paths)
            id_to_metric[uid] = (mse, w)
        except Exception as e:
            print(f"Error computing metric for id {uid}: {e}")
            id_to_metric[uid] = (float("inf"), None)

    # 4. Sort results by MSE (lower is better).
    sorted_results = sorted(id_to_metric.items(), key=lambda x: x[1][0])

    # Prepare summary data: id, mse, a, b
    summary_data = []
    for uid, (mse, w) in sorted_results:
        if w is not None:
            a, b = w[0], w[1]
        else:
            a, b = "", ""
        summary_data.append((uid, mse, a, b))

    summary_csv = os.path.join(output_dir, "ellipsis_summary.csv")
    save_to_csv(summary_data, ["id", "mse", "a", "b"], summary_csv)
    print(f"Saved summary CSV to {summary_csv}")

    # 5. Save top_x scores (lowest MSE) to a CSV.
    top_scores = (
        sorted_results[:top_x] if len(sorted_results) >= top_x else sorted_results
    )
    top_scores_data = [
        (uid, mse, (w[0] if w is not None else ""), (w[1] if w is not None else ""))
        for uid, (mse, w) in top_scores
    ]
    top_csv = os.path.join(output_dir, f"ellipsis_top_{top_x}_scores.csv")
    save_to_csv(top_scores_data, ["id", "mse", "a", "b"], top_csv)
    print(f"Saved top {top_x} scores CSV to {top_csv}")

    # 6. Save bottom_x scores (highest MSE) to a CSV.
    bottom_scores = (
        sorted_results[-top_x:] if len(sorted_results) >= top_x else sorted_results
    )
    bottom_scores_data = [
        (uid, mse, (w[0] if w is not None else ""), (w[1] if w is not None else ""))
        for uid, (mse, w) in bottom_scores
    ]
    bottom_csv = os.path.join(output_dir, f"ellipsis_bottom_{top_x}_scores.csv")
    save_to_csv(bottom_scores_data, ["id", "mse", "a", "b"], bottom_csv)
    print(f"Saved bottom {top_x} scores CSV to {bottom_csv}")

    # 7. Compute median and its immediate neighbors.
    n = len(sorted_results)
    if n == 0:
        print("No valid files processed.")
        return
    if n % 2 == 1:
        median_index = n // 2
        median_item = sorted_results[median_index]
        neighbors = []
        if median_index - 1 >= 0:
            neighbors.append(sorted_results[median_index - 1])
        neighbors.append(median_item)
        if median_index + 1 < n:
            neighbors.append(sorted_results[median_index + 1])
        median_neighbors = neighbors
    else:
        median_index = (n // 2) - 1
        median_item1 = sorted_results[median_index]
        median_item2 = sorted_results[median_index + 1]
        neighbors = []
        if median_index - 1 >= 0:
            neighbors.append(sorted_results[median_index - 1])
        neighbors.extend([median_item1, median_item2])
        if median_index + 2 < n:
            neighbors.append(sorted_results[median_index + 2])
        median_neighbors = neighbors

    median_data = []
    for uid, (mse, w) in median_neighbors:
        if w is not None:
            a, b = w[0], w[1]
        else:
            a, b = "", ""
        median_data.append((uid, mse, a, b))
    median_csv = os.path.join(output_dir, "ellipsis_median_and_neighbors.csv")
    save_to_csv(median_data, ["id", "mse", "a", "b"], median_csv)
    print(f"Saved median and neighbors CSV to {median_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ellipsis metric for audio files."
    )
    parser.add_argument(
        "--separated_audio_dir",
        type=str,
        default="C:\\Users\\arifr\\git\\CommonVoice_RIR\\src\\evaluation_utils\\separation_output_prev",
        help="Path to the directory containing separated WAV files.",
    )
    parser.add_argument(
        "--mixtures_audio_dir",
        type=str,
        default="C:\\Users\\arifr\\git\\CommonVoice_RIR\\src\\evaluation_utils\\separation_input_prev",
        help="Path to the directory containing mixture WAV files. Files should be named {unique_id}.wav",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        nargs="*",
        default=[""],
        help="List of possible prefixes to remove from separated filenames.",
    )
    parser.add_argument(
        "--suffixes",
        type=str,
        nargs="*",
        default=["_spk1_corrected", "_spk2_corrected"],
        help="List of possible suffixes to remove from separated filenames.",
    )
    parser.add_argument(
        "--wav_ends_with",
        type=str,
        default="_corrected.wav",
        help="Process only separated files that end with this string.",
    )
    parser.add_argument(
        "--top_x",
        type=int,
        default=3,
        help="Number of top and bottom scores to output.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:\\Users\\arifr\\git\\CommonVoice_RIR\\src\\evaluation_utils\\unsupervised_metrics\\summary_elipsis_metric",
        help="Directory to save CSV files. If not provided, defaults to the separated_audio_dir.",
    )
    args = parser.parse_args()

    prefixes = process_arg_list(args.prefixes)
    suffixes = process_arg_list(args.suffixes)
    output_dir = args.output_dir if args.output_dir else args.separated_audio_dir

    main(
        args.separated_audio_dir,
        args.mixtures_audio_dir,
        prefixes,
        suffixes,
        wav_ends_with=args.wav_ends_with,
        top_x=args.top_x,
        output_dir=output_dir,
    )
