import pandas as pd
import random
import argparse

def generate_speaker_pairs(csv_path, output_csv, utterance_count_csv, speaker_pair_count_csv):
    """
    Generates a CSV of paired utterances, ensuring minimal pair repetition between unique speakers.
    Each speaker is paired with different speakers as evenly as possible, and utterances are not reused.
    Also tracks how many times each utterance was used and how many times each speaker pair was created.
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Create unique speaker identifiers
    df["Unique_Speaker"] = df["Gender"] + "_" + df["Speaker"]
    
    # Group utterances by speaker and shuffle
    speaker_groups = {sp: group.sample(frac=1, random_state=42).to_dict("records") for sp, group in df.groupby("Unique_Speaker")}
    
    # Extract unique speakers
    unique_speakers = list(speaker_groups.keys())
    
    # Initialize pairing storage
    utterance_pairs = []
    speaker_pair_counts = {sp: {} for sp in unique_speakers}  # Track pair counts between speakers
    utterance_usage_counts = {}  # Track how many times each utterance is used
    
    while any(speaker_groups.values()):
        random.shuffle(unique_speakers)  # Shuffle speakers each iteration
        used_speakers = set()
        
        for speaker in unique_speakers:
            if not speaker_groups[speaker] or speaker in used_speakers:
                continue
            
            available_partners = [sp for sp in unique_speakers if sp != speaker and speaker_groups.get(sp) and sp not in used_speakers]
            
            if not available_partners:
                continue
            
            # Choose the partner with the fewest existing pairings
            partner = min(available_partners, key=lambda sp: speaker_pair_counts[speaker].get(sp, 0))
            
            # Get one utterance from each
            utterance_1 = speaker_groups[speaker].pop(0)
            utterance_2 = speaker_groups[partner].pop(0)
            
            # Store the pair
            utterance_pairs.append({
                "Speaker_1": speaker,
                "Speaker_2": partner,
                "Utterance_1": utterance_1["Utterance_Path"],
                "Utterance_2": utterance_2["Utterance_Path"],
                "Gender_1": utterance_1["Gender"],
                "Gender_2": utterance_2["Gender"],
                "Session_1": utterance_1["Session"],
                "Session_2": utterance_2["Session"],
                "SubCategory_1": utterance_1["SubCategory"],
                "SubCategory_2": utterance_2["SubCategory"],
                "Recording_Type_1": utterance_1["Recording_Type"],
                "Recording_Type_2": utterance_2["Recording_Type"]
            })
            
            # Update pairing counts
            speaker_pair_counts[speaker][partner] = speaker_pair_counts[speaker].get(partner, 0) + 1
            speaker_pair_counts[partner][speaker] = speaker_pair_counts[partner].get(speaker, 0) + 1
            
            # Track utterance usage counts
            utterance_usage_counts[utterance_1["Utterance_Path"]] = utterance_usage_counts.get(utterance_1["Utterance_Path"], 0) + 1
            utterance_usage_counts[utterance_2["Utterance_Path"]] = utterance_usage_counts.get(utterance_2["Utterance_Path"], 0) + 1
            
            # Mark as used in this round
            used_speakers.update([speaker, partner])
    
    # Save paired utterances to CSV
    df_output = pd.DataFrame(utterance_pairs)
    df_output.to_csv(output_csv, index=False)
    
    # Save utterance usage counts to CSV
    df_utterance_counts = pd.DataFrame(utterance_usage_counts.items(), columns=["Utterance_Path", "Usage_Count"])
    df_utterance_counts.to_csv(utterance_count_csv, index=False)
    
    # Save speaker pair counts to CSV
    speaker_pair_list = [(sp1, sp2, count) for sp1 in speaker_pair_counts for sp2, count in speaker_pair_counts[sp1].items() if sp1 < sp2]
    df_speaker_pair_counts = pd.DataFrame(speaker_pair_list, columns=["Speaker_1", "Speaker_2", "Pair_Count"])
    df_speaker_pair_counts.to_csv(speaker_pair_count_csv, index=False)

def main():
    parser = argparse.ArgumentParser(description="Generate speaker utterance pairs from a dataset.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file.")
    parser.add_argument("--utterance_count_csv", type=str, required=True, help="Path to output utterance usage count CSV file.")
    parser.add_argument("--speaker_pair_count_csv", type=str, required=True, help="Path to output speaker pair count CSV file.")
    
    args = parser.parse_args()
    
    generate_speaker_pairs(args.input_csv, args.output_csv, args.utterance_count_csv, args.speaker_pair_count_csv)
    print(f"Generated utterance pairs saved to {args.output_csv}")
    print(f"Generated utterance usage counts saved to {args.utterance_count_csv}")
    print(f"Generated speaker pair counts saved to {args.speaker_pair_count_csv}")

if __name__ == "__main__":
    main()
