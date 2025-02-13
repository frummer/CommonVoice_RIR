#!/usr/bin/env python3

"""
balance_pairs_script.py

This script takes an input CSV containing rows of:
    mic, unique_speaker_id_1, path_1, unique_speaker_id_2, path_2

It creates a balanced subset where:
  - Each pair of speakers is represented by the SAME number of utterances.
  - The total number of rows does not exceed the user-specified 'mixtures_amount'.
  
Additionally, it writes a summary CSV that shows how many utterances each pair has
in the final balanced output.

Typical usage:
    python balance_pairs_script.py \
        --csv_in=input.csv \
        --csv_out=output.csv \
        --summary_out=pairs_summary.csv \
        --mixtures_amount=1000 \
        --random_state=42
"""

import argparse

import pandas as pd


def balance_pairs_with_limit_and_summary(
    csv_in: str,
    csv_out: str,
    mixtures_amount: int,
    summary_out: str,
    random_state: int = 42,
) -> None:
    """
    Reads an input CSV with columns:
       [mic, unique_speaker_id_1, path_1, unique_speaker_id_2, path_2]
    and writes out a new CSV where:
      - Each pair of speakers is represented by the SAME number of utterances.
      - The total number of rows is at most 'mixtures_amount'.

    Additionally, writes a separate 'summary_out' CSV listing each pair_id
    and the count of utterances that pair has in the final balanced CSV.

    Parameters
    ----------
    csv_in : str
        Path to the input CSV file.
    csv_out : str
        Path to the output CSV file (balanced dataset).
    mixtures_amount : int
        Desired maximum total number of rows in the balanced output.
    summary_out : str
        Path to the summary CSV file, which will show each pair and the
        number of utterances assigned to it.
    random_state : int, optional
        Random seed for sampling, by default 42.

    Returns
    -------
    None
        This function writes the balanced CSV and the summary CSV to disk.
    """

    # Read the original CSV
    df = pd.read_csv(csv_in)

    # Create a canonical pair ID so (A,B) == (B,A).
    # If you want them distinct, remove 'sorted()'.
    def make_pair_id(row):
        spk1 = row["unique_speaker_id_1"]
        spk2 = row["unique_speaker_id_2"]
        return "__".join(sorted([spk1, spk2]))

    df["pair_id"] = df.apply(make_pair_id, axis=1)

    # Group by pair_id
    grouped = df.groupby("pair_id")

    # Count how many rows per pair
    pair_counts = grouped.size()
    n_pairs = grouped.ngroups

    # Find the minimum group size among all pairs
    min_group_size = pair_counts.min()

    # Compute how many rows we want per pair:
    #   total desired / number_of_pairs
    per_pair = mixtures_amount // n_pairs

    # We cannot exceed the smallest pair’s size
    per_pair = min(per_pair, min_group_size)

    # If per_pair is zero, no balanced subset can be created
    if per_pair == 0:
        print("[Warning] 'per_pair' is zero; no balanced subset can be created.")
        # Write an empty CSV or handle as you see fit:
        df.head(0).to_csv(csv_out, index=False)
        summary_df = pd.DataFrame(columns=["pair_id", "utterances_count"])
        summary_df.to_csv(summary_out, index=False)
        return

    # Sample from each group
    balanced_dfs = []
    for _, group_df in grouped:
        if len(group_df) > per_pair:
            balanced_dfs.append(group_df.sample(n=per_pair, random_state=random_state))
        else:
            balanced_dfs.append(group_df)

    # Combine all sampled pairs
    balanced_df = pd.concat(balanced_dfs, axis=0)

    # (Optional) Shuffle the final result
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )

    # Write the balanced data to CSV
    balanced_df.to_csv(csv_out, index=False)

    # ------------------------------
    # Create the summary CSV
    # ------------------------------
    final_grouped = (
        balanced_df.groupby("pair_id").size().reset_index(name="utterances_count")
    )
    final_grouped.to_csv(summary_out, index=False)

    final_count = len(balanced_df)
    print(
        f"Done. Each pair has {per_pair} utterances. "
        f"Total rows in output = {final_count} (<= {mixtures_amount})."
    )
    print(f"Summary file written to: {summary_out}")


def main():
    """
    Parse command-line arguments and run the balancing procedure.
    """
    parser = argparse.ArgumentParser(
        description="Balance a CSV so each speaker pair has the same number of utterances, "
        "up to the user-specified total (mixtures_amount). "
        "Also produces a summary CSV with pair-level counts."
    )

    parser.add_argument(
        "--csv_in",
        type=str,
        default="C:\\Users\\arifr\\git\\CommonVoice_RIR\\src_ldc\\v2\\test_pairs.csv",
        help="Path to the input CSV file (default: input.csv).",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default="output.csv",
        help="Path to the output CSV file (balanced) (default: output.csv).",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        default="pairs_summary.csv",
        help="Path to the summary CSV file listing pairs and their counts (default: pairs_summary.csv).",
    )
    parser.add_argument(
        "--mixtures_amount",
        type=int,
        default=12000,
        help="Desired maximum total number of rows in the balanced output (default: 1000).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )

    args = parser.parse_args()

    balance_pairs_with_limit_and_summary(
        csv_in=args.csv_in,
        csv_out=args.csv_out,
        mixtures_amount=args.mixtures_amount,
        summary_out=args.summary_out,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
