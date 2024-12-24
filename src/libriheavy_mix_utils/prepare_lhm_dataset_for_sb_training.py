import json
import os
import shutil


def copy_mixture_and_sources_by_id(
    jsonl_path,
    mixtures_dir,
    sources_dir,
    output_dir,
    folder_names=None,
    rename_sources=False,
):
    """
    Reads a .jsonl file line-by-line, extracts the 'id' (UUID) from each line,
    and copies:
      - The mixture file from mixtures_dir/<uuid>.flac
      - The two source files from sources_dir/<uuid>/0.flac & 1.flac

    into the following structure:
        output_dir/
            mixture/
            source1/
            source2/

    Args:
        jsonl_path (str):
            Path to the .jsonl file where each line is a JSON object with an 'id' key.
        mixtures_dir (str):
            Directory containing mixture files named '<uuid>.flac'.
        sources_dir (str):
            Directory with subfolders named '<uuid>/', each holding '0.flac' and '1.flac'.
        output_dir (str):
            Base directory where we copy the mixture and sources.
        folder_names (dict):
            Optional mapping for subfolder names. Defaults to:
                { "mixture": "mixture", "source1": "source1", "source2": "source2" }
        rename_sources (bool):
            If True, rename source files to something like '<uuid>_0.flac' and '<uuid>_1.flac'.
            If False, keep them as '0.flac' and '1.flac'.
    """

    if folder_names is None:
        folder_names = {
            "mixture": "mixture",
            "source1": "source1",
            "source2": "source2",
        }

    # Create the output subfolders
    mixture_outdir = os.path.join(output_dir, folder_names["mixture"])
    source1_outdir = os.path.join(output_dir, folder_names["source1"])
    source2_outdir = os.path.join(output_dir, folder_names["source2"])

    os.makedirs(mixture_outdir, exist_ok=True)
    os.makedirs(source1_outdir, exist_ok=True)
    os.makedirs(source2_outdir, exist_ok=True)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip empty lines

            # Parse JSON from this line
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON decode error on line #{line_idx}: {e}")
                continue

            # We only need the 'id' field from each JSON line
            mixture_id = data.get("id")
            if not mixture_id:
                print(f"[Warning] No 'id' key found in line #{line_idx}. Skipping.")
                continue

            # 1) Copy the mixture file from mixtures_dir/<uuid>.flac
            mixture_path = os.path.join(mixtures_dir, f"{mixture_id}.flac")
            if not os.path.isfile(mixture_path):
                print(
                    f"[Warning] Mixture file not found for ID '{mixture_id}' at {mixture_path}. Skipping."
                )
                continue

            mixture_target = os.path.join(mixture_outdir, f"{mixture_id}.flac")
            shutil.copy2(mixture_path, mixture_target)
            print(f"[INFO] Copied mixture: {mixture_path} -> {mixture_target}")

            # 2) Copy the source files from sources_dir/<uuid>/0.flac and 1.flac
            source_folder = os.path.join(sources_dir, mixture_id)
            source0 = os.path.join(source_folder, "0.flac")
            source1 = os.path.join(source_folder, "1.flac")

            if not (os.path.isfile(source0) and os.path.isfile(source1)):
                print(
                    f"[Warning] Could not find '0.flac' and/or '1.flac' in {source_folder} for '{mixture_id}'."
                )
                continue

            # Decide the target filenames
            if rename_sources:
                # e.g., "510c8613-0aec-4cf7-9720-2f07a4f69e24_0.flac"
                src0_name = f"{mixture_id}.flac"
                src1_name = f"{mixture_id}.flac"
            else:
                # keep them as "0.flac" and "1.flac"
                src0_name = "0.flac"
                src1_name = "1.flac"

            src0_target = os.path.join(source1_outdir, src0_name)
            src1_target = os.path.join(source2_outdir, src1_name)

            shutil.copy2(source0, src0_target)
            print(f"[INFO] Copied source0: {source0} -> {src0_target}")

            shutil.copy2(source1, src1_target)
            print(f"[INFO] Copied source1: {source1} -> {src1_target}")


if __name__ == "__main__":
    # Example usage:
    jsonl_path = "C:\\Users\\arifr\\git\\CommonVoice_RIR\\sample_2_spk.jsonl"
    dataset_type = "train"
    mixtures_dir = "C:\\Users\\arifr\\LibriheavyMix-small\\audio_anechoic"
    sources_dir = "C:\\Users\\arifr\\LibriheavyMix-small\\src_anechoic"
    output_dir = f"C:\\Users\\arifr\\git\\CommonVoice_RIR\\lhm_2spk\\{dataset_type}"
    folder_map = {"mixture": "mixture", "source1": "source1", "source2": "source2"}

    # If True, rename source files to <uuid>_0.flac and <uuid>_1.flac.
    # If False, keep them as 0.flac and 1.flac.
    rename_sources = True

    copy_mixture_and_sources_by_id(
        jsonl_path=jsonl_path,
        mixtures_dir=mixtures_dir,
        sources_dir=sources_dir,
        output_dir=output_dir,
        folder_names=folder_map,
        rename_sources=rename_sources,
    )
