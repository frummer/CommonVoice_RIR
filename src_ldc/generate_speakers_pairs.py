import pandas as pd
from itertools import combinations, cycle
import random
import argparse

def generate_speaker_pairs(csv_path):
    """
    Generates speaker utterance pairs from a given dataset, ensuring balanced pairing across speakers.

    Args:
        csv_path (str): Path to the input CSV file containing utterance data.

    Input CSV format:
        - Columns:
            - Gender: Gender of the speaker (e.g., Male, Female).
            - Session: Session identifier.
            - Speaker: Speaker identifier.
            - SubCategory: Type of utterance.
            - Recording_Type: Recording setup used.
            - Utterance_Path: Path to the recorded utterance file.
        - Example row:
            | Gender | Session   | Speaker | SubCategory         | Recording_Type | Utterance_Path |
            |--------|-----------|---------|---------------------|----------------|----------------|
            | Female | Session_1 | NS1     | 1.SAAB_sentences_1  | Yamaha_Mixer   | path/to/file1.wav |

    Output CSV formats:
        1. Paired utterances CSV:
            - Columns:
                - Speaker_1, Speaker_2: Unique speaker identifiers.
                - Utterance_1, Utterance_2: Paths to the paired utterances.
                - Gender_1, Gender_2: Gender of each speaker.
                - Session_1, Session_2: Session information.
                - SubCategory_1, SubCategory_2: Utterance subcategories.
                - Recording_Type_1, Recording_Type_2: Recording type.
            - Example row:
                | Speaker_1 | Speaker_2 | Utterance_1 | Utterance_2 | Gender_1 | Gender_2 | Session_1 | Session_2 | SubCategory_1 | SubCategory_2 | Recording_Type_1 | Recording_Type_2 |
                |-----------|-----------|-------------|-------------|----------|----------|-----------|-----------|---------------|---------------|-----------------|-----------------|
                | F_NS1     | M_NS2     | path1.wav   | path2.wav   | Female   | Male     | S1        | S2        | Category1     | Category2     | Type1           | Type2           |
        
        2. Speaker pair counts CSV:
            - Columns:
                - Speaker_1, Speaker_2: Unique speaker identifiers.
                - Pair_Count: Number of paired utterances.
            - Example row:
                | Speaker_1 | Speaker_2 | Pair_Count |
                |-----------|-----------|------------|
                | F_NS1     | M_NS2     | 10         |
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Create unique speaker identifiers considering gender
    df["Unique_Speaker"] = df["Gender"] + "_" + df["Speaker"]
    
    # Extract unique speakers
    unique_speakers = df["Unique_Speaker"].unique()
    random.shuffle(unique_speakers)  # Shuffle for randomness
    
    # Create a cycle iterator to ensure diverse pairing
    speaker_cycle = cycle(unique_speakers)
    
    # Create utterance pairs with different speakers
    utterance_pairs = []
    used_utterances = set()
    pair_counts = {}
    
    for speaker in unique_speakers:
        df_speaker = df[df["Unique_Speaker"] == speaker]
        
        for _, utterance in df_speaker.iterrows():
            # Find the next different speaker
            paired_speaker = next(speaker_cycle)
            while paired_speaker == speaker:
                paired_speaker = next(speaker_cycle)
            
            df_paired_speaker = df[df["Unique_Speaker"] == paired_speaker]
            
            # Select an unused utterance from the paired speaker
            df_paired_speaker = df_paired_speaker[~df_paired_speaker["Utterance_Path"].isin(used_utterances)]
            if df_paired_speaker.empty:
                continue
            paired_utterance = df_paired_speaker.sample(n=1, random_state=42).iloc[0]
            
            # Mark utterances as used
            used_utterances.add(utterance["Utterance_Path"])
            used_utterances.add(paired_utterance["Utterance_Path"])
            
            # Store the pair
            utterance_pairs.append({
                "Speaker_1": speaker,
                "Speaker_2": paired_speaker,
                "Utterance_1": utterance["Utterance_Path"],
                "Utterance_2": paired_utterance["Utterance_Path"],
                "Gender_1": utterance["Gender"],
                "Gender_2": paired_utterance["Gender"],
                "Session_1": utterance["Session"],
                "Session_2": paired_utterance["Session"],
                "SubCategory_1": utterance["SubCategory"],
                "SubCategory_2": paired_utterance["SubCategory"],
                "Recording_Type_1": utterance["Recording_Type"],
                "Recording_Type_2": paired_utterance["Recording_Type"]
            })
            
            # Track pairing counts
            pair_key = tuple(sorted([speaker, paired_speaker]))
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
    
    # Handle remaining unpaired utterances
    remaining_utterances = df[~df["Utterance_Path"].isin(used_utterances)]
    remaining_list = remaining_utterances.sample(frac=1, random_state=42).to_dict("records")
    
    for i in range(0, len(remaining_list) - 1, 2):
        utterance_pairs.append({
            "Speaker_1": remaining_list[i]["Unique_Speaker"],
            "Speaker_2": remaining_list[i + 1]["Unique_Speaker"],
            "Utterance_1": remaining_list[i]["Utterance_Path"],
            "Utterance_2": remaining_list[i + 1]["Utterance_Path"],
            "Gender_1": remaining_list[i]["Gender"],
            "Gender_2": remaining_list[i + 1]["Gender"],
            "Session_1": remaining_list[i]["Session"],
            "Session_2": remaining_list[i + 1]["Session"],
            "SubCategory_1": remaining_list[i]["SubCategory"],
            "SubCategory_2": remaining_list[i + 1]["SubCategory"],
            "Recording_Type_1": remaining_list[i]["Recording_Type"],
            "Recording_Type_2": remaining_list[i + 1]["Recording_Type"]
        })
        
        # Track pairing counts
        pair_key = tuple(sorted([remaining_list[i]["Unique_Speaker"], remaining_list[i + 1]["Unique_Speaker"]]))
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
    
    # Convert to DataFrame
    pairs_df = pd.DataFrame(utterance_pairs)
    
    # Convert pair counts to DataFrame
    pair_counts_df = pd.DataFrame([(p[0], p[1], count) for p, count in pair_counts.items()], 
                                  columns=["Speaker_1", "Speaker_2", "Pair_Count"])
    
    return pairs_df, pair_counts_df

def main():
    parser = argparse.ArgumentParser(description="Generate speaker utterance pairs from a dataset.")
    parser.add_argument("--input_csv", type=str, default="/home/afrumme1/CommonVoice_RIR/csv_output/utterance_mapping.csv", help="Path to input CSV file.")
    parser.add_argument("--output_pairs_csv", type=str, default="/home/afrumme1/CommonVoice_RIR/csv_output/output_pairs.csv", help="Path to output pairs CSV file.")
    parser.add_argument("--output_counts_csv", type=str, default="/home/afrumme1/CommonVoice_RIR/csv_output/output_counts.csv", help="Path to output pairs count CSV file.")
    
    args = parser.parse_args()
    
    pairs_df, pair_counts_df = generate_speaker_pairs(args.input_csv)
    
    # Save the outputs
    pairs_df.to_csv(args.output_pairs_csv, index=False)
    pair_counts_df.to_csv(args.output_counts_csv, index=False)
    
    print(f"Generated utterance pairs saved to {args.output_pairs_csv}")
    print(f"Generated speaker pair counts saved to {args.output_counts_csv}")

if __name__ == "__main__":
    main()
