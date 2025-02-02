import os
import csv
import argparse
from tqdm import tqdm

def scan_dataset(root_dir, output_dir, csv_filename, room_keyword, recording_filter=None):
    """
    Scans the dataset directory and writes a CSV file mapping each utterance to its metadata.
    
    Expected directory structure:
    
      dataset_root/
        ├── Females
        │    ├── Session_1
        │    │    ├── Speaker_X (e.g., NS1)
        │    │    │    └── 2.Silent_Room   (or a folder name ending with "Silent_Room")
        │    │    │          ├── 10.Questions_1
        │    │    │          │      └── *.flac
        │    │    │          ├── 1.SAAB_sentences_1
        │    │    │          │      └── *.flac
        │    │    │          └── ... (other subfolders)
        │    │    └── ... (other speakers)
        │    └── ... (other sessions)
        └── Males
             └── ... (similar structure)
    
    Parameters:
      root_dir (str): Root directory of the dataset.
      output_dir (str): Directory where the CSV file will be saved.
      csv_filename (str): Output CSV file name.
      room_keyword (str): Keyword to identify the silent room folder.
      recording_filter (str, optional): If provided, only include FLAC files whose names contain
                                        this substring (case-insensitive).
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, csv_filename)
    csv_rows = []
    genders = ["Females", "Males"]

    for gender in tqdm(genders, desc="Processing genders"):
        gender_path = os.path.join(root_dir, gender)
        if not os.path.isdir(gender_path):
            print(f"Warning: {gender_path} is not a valid directory. Skipping.")
            continue

        sessions = sorted(os.listdir(gender_path)) if os.path.isdir(gender_path) else []
        for session in tqdm(sessions, desc=f"Processing {gender}", leave=False, disable=len(sessions) == 0):
            session_path = os.path.join(gender_path, session)
            if not os.path.isdir(session_path):
                continue

            speakers = sorted(os.listdir(session_path), key=lambda x: (x[0].isalpha(), int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)) if os.path.isdir(session_path) else []
            for speaker in tqdm(speakers, desc=f"Processing {session}", leave=False, disable=len(speakers) == 0):
                speaker_path = os.path.join(session_path, speaker)
                if not os.path.isdir(speaker_path):
                    continue

                silent_room_path = None
                for folder in sorted(os.listdir(speaker_path)):
                    folder_path = os.path.join(speaker_path, folder)
                    if os.path.isdir(folder_path) and room_keyword.lower() in folder.lower():
                        silent_room_path = folder_path
                        break

                if not silent_room_path:
                    print(f"Warning: No '{room_keyword}' folder found for speaker {speaker} in {speaker_path}. Skipping.")
                    continue

                subcats = sorted(os.listdir(silent_room_path)) 
                for subcat in tqdm(subcats, desc=f"Processing {speaker}", leave=False, disable=len(subcats) == 0):
                    subcat_path = os.path.join(silent_room_path, subcat)
                    if not os.path.isdir(subcat_path):
                        continue

                    files = sorted(os.listdir(subcat_path)) 
                    for file in tqdm(files, desc=f"Processing {subcat}", leave=False, disable=len(files) == 0):
                        if not file.lower().endswith(".flac"):
                            continue

                        if recording_filter and recording_filter.lower() not in file.lower():
                            continue

                        file_path = os.path.join(subcat_path, file)
                        recording_type = os.path.splitext(file)[0]

                        csv_rows.append([gender, session, speaker, subcat, recording_type, file_path])

    with open(output_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Gender", "Session", "Speaker", "SubCategory", "Recording_Type", "Utterance_Path"])
        writer.writerows(csv_rows)

    print(f"CSV file '{output_path}' created with {len(csv_rows)} utterance entries.")

def main():
    parser = argparse.ArgumentParser(description="Scan dataset and create CSV mapping for utterances.")
    parser.add_argument("--dataset_root", type=str, default="/export/corpora5/LDC/LDC2014S02/data", help="Root directory of the dataset")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory where the output CSV file will be saved")
    parser.add_argument("--csv_filename", type=str, default="utterance_mapping.csv", help="Output CSV file name")
    parser.add_argument("--room", type=str, default="Silent_Room", help="Keyword to identify the silent room folder")
    parser.add_argument("--recording_filter", type=str, default="Computer_Mic_Front", help="Optional recording filter (e.g., 'Computer_Mic_Front')")

    args = parser.parse_args()
    scan_dataset(args.dataset_root, args.output_dir, args.csv_filename, args.room, recording_filter=args.recording_filter)

if __name__ == '__main__':
    main()
