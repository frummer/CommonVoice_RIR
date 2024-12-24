import json


def main(dataset_json_path: str, output_filtered_dataset_path: str):
    total_rows = 0
    filtered_rows = []

    # 1. Read the .jsonl file line by line
    with open(dataset_json_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            # Parse this line as a single JSON object
            row = json.loads(line)
            total_rows += 1

            # 2. Collect unique speakers
            supervisions = row.get("supervisions", [])
            speakers = set()
            for sup in supervisions:
                if "speaker" in sup:
                    speakers.add(sup["speaker"])

            # 3. If we have exactly 2 unique speakers, keep it
            if len(speakers) == 2:
                filtered_rows.append(row)

    # 4. Write filtered rows back as .jsonl, one JSON object per line
    with open(output_filtered_dataset_path, "w", encoding="utf-8") as f_out:
        for row in filtered_rows:
            json.dump(row, f_out, ensure_ascii=False)
            f_out.write("\n")

    print(f"Original rows: {total_rows}")
    print(f"Filtered rows: {len(filtered_rows)}")


if __name__ == "__main__":
    path_to_dataset_json = "C:\\Users\\arifr\\git\\CommonVoice_RIR\\lsheavymix_cuts_train_small_snr_aug_mono_rir_fixed.jsonl"
    path_to_filtered_dataset_json = "C:\\Users\\arifr\\git\\CommonVoice_RIR\\2_spk_lsheavymix_cuts_train_small_snr_aug_mono_rir_fixed.jsonl"
    main(
        dataset_json_path=path_to_dataset_json,
        output_filtered_dataset_path=path_to_filtered_dataset_json,
    )
