#!/usr/bin/env python3
"""
Merge three CSV files into a single CSV by matching on `id` column.

- First CSV (no header assumed): columns = id, mse, a, b
  * We'll rename 'mse' to 'mixit_mse'.

- Second CSV (no header assumed): columns = id, score, audio1_start_sec, audio1_end_sec, audio2_start_sec, audio2_end_sec, top_entries
  * We'll rename 'score' to 'leakage_max_score'.

- Third CSV (no header assumed): columns = id, score
  * We'll rename 'score' to 'unseparation_correlation'.

All three DataFrames are then merged on `id` using an "outer" join (i.e., include all ids
found in any of the files).

Example usage:
    python merge_csv.py --file1 file1.csv --file2 file2.csv --file3 file3.csv --output summarized_scores.csv
"""

import argparse
import pandas as pd

def do_merge(file1, file2, file3, output,
             cols_file1, cols_file2, cols_file3,
             rename_mse, rename_leakage_score, rename_unsep_corr):
    """
    1) Read the three CSV files with assigned column names.
    2) Rename the designated score columns.
    3) Merge them by `id` using an outer join.
    4) Write the merged DataFrame to a new CSV.
    """
    # Read first CSV and rename 'mse' -> rename_mse
    df1 = pd.read_csv(file1, header=None, names=cols_file1)
    df1 = df1.rename(columns={"mse": rename_mse})
    
    # Read second CSV and rename 'score' -> rename_leakage_score
    df2 = pd.read_csv(file2, header=None, names=cols_file2)
    df2 = df2.rename(columns={"score": rename_leakage_score})
    
    # Read third CSV and rename 'score' -> rename_unsep_corr
    df3 = pd.read_csv(file3, header=None, names=cols_file3)
    df3 = df3.rename(columns={"score": rename_unsep_corr})
    
    # Merge DataFrames (outer join to keep all IDs)
    merged = df1.merge(
        df2[["id", rename_leakage_score]],
        on="id",
        how="outer"
    ).merge(
        df3[["id", rename_unsep_corr]],
        on="id",
        how="outer"
    )

    # Define final column order
    final_cols = [
        "id",
        rename_mse,  # i.e., 'mixit_mse'
        "a",
        "b",
        rename_leakage_score,         # i.e., 'leakage_max_score'
        rename_unsep_corr             # i.e., 'unseparation_correlation'
    ]
    
    # Write to CSV
    merged[final_cols].to_csv(output, index=False)
    print(f"Merged CSV written to {output}")


def main():
    parser = argparse.ArgumentParser(description="Merge three CSV files on 'id' and rename score columns.")
    parser.add_argument(
        "--file1",
        default="C:\\Users\\arifr\\git\\CommonVoice_RIR\\output_dir\\LDC_V1\\ellipsis\\ellipsis_summary.csv",
        help="Path to the first CSV - ellipsis summary(default: file1.csv)."
    )
    parser.add_argument(
        "--file2",
        default="C:\\Users\\arifr\\git\\CommonVoice_RIR\\output_dir\\LDC_V1\\leakage\\summary\\all_scores.csv",
        help="Path to the second CSV - Leakage summary(default: file2.csv)."
    )
    parser.add_argument(
        "--file3",
        default="C:\\Users\\arifr\\git\\CommonVoice_RIR\\output_dir\\LDC_V1\\unseparation\\all_scores.csv",
        help="Path to the third CSV - unseparation summary(default: file3.csv)."
    )
    parser.add_argument(
        "--output",
        default="summarized_metrics_scores.csv",
        help="Name of the output CSV (default: summarized_scores.csv)."
    )
    args = parser.parse_args()

    # Define constants (column layouts, rename targets)
    COLS_FILE1 = ["id", "mse", "a", "b"]
    COLS_FILE2 = ["id", "score", 
                  "audio1_start_sec", "audio1_end_sec", 
                  "audio2_start_sec", "audio2_end_sec", 
                  "top_entries"]
    COLS_FILE3 = ["id", "score"]

    MSE_RENAMED = "mixit_mse"
    LEAKAGE_MAX_SCORE_RENAMED = "leakage_max_score"
    UNSEPARATION_CORRELATION_RENAMED = "unseparation_correlation"

    # Perform the merge
    do_merge(
        file1=args.file1,
        file2=args.file2,
        file3=args.file3,
        output=args.output,
        cols_file1=COLS_FILE1,
        cols_file2=COLS_FILE2,
        cols_file3=COLS_FILE3,
        rename_mse=MSE_RENAMED,
        rename_leakage_score=LEAKAGE_MAX_SCORE_RENAMED,
        rename_unsep_corr=UNSEPARATION_CORRELATION_RENAMED
    )

if __name__ == "__main__":
    main()
