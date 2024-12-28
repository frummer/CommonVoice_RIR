import json
import logging
import os

import librosa
import soundfile as sf


def copy_and_convert_audio_by_id_librosa(
    jsonl_path,
    mixtures_dir,
    sources_dir,
    output_dir,
    folder_names=None,
    rename_sources=False,
    target_sr=None,
    overwrite=False,  # New parameter to control overwriting
):
    """
    Copies and converts audio files based on IDs from a JSONL file.

    Parameters:
        jsonl_path (str): Path to the JSONL file containing audio IDs.
        mixtures_dir (str): Directory containing mixture .flac files named <id>.flac.
        sources_dir (str): Directory containing source subdirectories named by <id>, each with 0.flac, 1.flac, 0_padded.flac, and/or 1_padded.flac.
        output_dir (str): Directory where converted .wav files will be saved.
        folder_names (dict, optional): Mapping for output subfolder names.
                                       Defaults to {'mixture': 'mixture', 'source1': 'source1', 'source2': 'source2'}.
        rename_sources (bool, optional): If True, source files are renamed to <id>_0.wav, <id>_1.wav, etc.
                                         If a padded file is used, the `_padded` suffix is retained.
                                         Defaults to False.
        target_sr (int, optional): Target sample rate for conversion. If None, original sample rate is used.
        overwrite (bool, optional): If True, existing files will be overwritten. Defaults to False.
    """

    if folder_names is None:
        folder_names = {
            "mixture": "mixture",
            "source1": "source1",
            "source2": "source2",
        }

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("conversion.log"), logging.StreamHandler()],
    )

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
                logging.warning(f"JSON decode error on line #{line_idx}: {e}")
                continue

            # We only need the 'id' field from each JSON line
            mixture_id = data.get("id")
            if not mixture_id:
                logging.warning(f"No 'id' key found in line #{line_idx}. Skipping.")
                continue

            # 1) Convert the mixture file from mixtures_dir/<uuid>.flac to WAV
            mixture_path = os.path.join(mixtures_dir, f"{mixture_id}.flac")
            if not os.path.isfile(mixture_path):
                logging.warning(
                    f"Mixture file not found for ID '{mixture_id}' at {mixture_path}. Skipping."
                )
                continue

            mixture_target = os.path.join(mixture_outdir, f"{mixture_id}.wav")

            if not overwrite and os.path.exists(mixture_target):
                logging.info(
                    f"Mixture target file already exists: {mixture_target}. Skipping."
                )
            else:
                try:
                    # Load the mixture audio using librosa
                    data_mixture, sr_mixture = librosa.load(
                        mixture_path, sr=target_sr, mono=False
                    )

                    # If multi-channel, librosa returns shape (channels, samples)
                    # soundfile.write expects (samples, channels), so transpose
                    if data_mixture.ndim > 1:
                        data_mixture = data_mixture.T

                    # Write the mixture audio to WAV using soundfile
                    sf.write(mixture_target, data_mixture, sr_mixture)
                    logging.info(
                        f"Converted and saved mixture: {mixture_path} -> {mixture_target}"
                    )
                except Exception as e:
                    logging.error(f"Failed to convert mixture '{mixture_id}': {e}")
                    continue

            # 2) Convert the source files from sources_dir/<uuid>/0.flac, 1.flac, 0_padded.flac, and/or 1_padded.flac to WAV
            source_folder = os.path.join(sources_dir, mixture_id)
            if not os.path.isdir(source_folder):
                logging.warning(
                    f"Source folder does not exist: {source_folder}. Skipping ID '{mixture_id}'."
                )
                continue

            # Define possible source filenames
            sources = {
                "0": ["0_padded.flac", "0.flac"],
                "1": ["1_padded.flac", "1.flac"],
            }

            for key, filenames in sources.items():
                source_used = False
                for filename in filenames:
                    source_path = os.path.join(source_folder, filename)
                    if os.path.isfile(source_path):
                        # Decide the target filename
                        if rename_sources:
                            target_name = f"{mixture_id}.wav"  # e.g., '12345_0.wav'
                        else:
                            target_name = f"{filename.split('.')[0]}.wav"  # ' '0.wav'

                        # Determine target directory based on key
                        if key == "0":
                            src_target = os.path.join(source1_outdir, target_name)
                        elif key == "1":
                            src_target = os.path.join(source2_outdir, target_name)
                        else:
                            logging.warning(
                                f"Unknown source key '{key}' for '{filename}'. Skipping."
                            )
                            continue

                        if not overwrite and os.path.exists(src_target):
                            logging.info(
                                f"Source target file already exists: {src_target}. Skipping."
                            )
                            source_used = True
                            break  # Skip to the next source
                        else:
                            try:
                                # Load the source audio using librosa
                                data_source, sr_source = librosa.load(
                                    source_path, sr=target_sr, mono=False
                                )
                                if data_source.ndim > 1:
                                    data_source = data_source.T
                                # Write the source audio to WAV using soundfile
                                sf.write(src_target, data_source, sr_source)
                                logging.info(
                                    f"Converted and saved source '{filename}': {source_path} -> {src_target}"
                                )
                                source_used = True
                                break  # Source has been processed
                            except Exception as e:
                                logging.error(
                                    f"Failed to convert source '{filename}' for '{mixture_id}': {e}"
                                )
                                continue
                if not source_used:
                    logging.warning(
                        f"No valid source files found for key '{key}' in '{source_folder}' for '{mixture_id}'. Skipping."
                    )


if __name__ == "__main__":
    # Example usage:
    jsonl_path = "C:\\Users\\arifr\\git\\CommonVoice_RIR\\sample_2_spk.jsonl"
    dataset_type = "train"
    mixtures_dir = "C:\\Users\\arifr\\LibriheavyMix-small\\audio_anechoic"
    sources_dir = "C:\\Users\\arifr\\LibriheavyMix-small\\src"
    output_dir = f"C:\\Users\\arifr\\git\\CommonVoice_RIR\\lhm_2spk\\{dataset_type}"
    folder_map = {"mixture": "mixture", "source1": "source1", "source2": "source2"}

    rename_sources = True
    target_sample_rate = None  # or set to desired sample rate, e.g., 44100
    overwrite_existing = False  # Set to True to overwrite existing files

    copy_and_convert_audio_by_id_librosa(
        jsonl_path=jsonl_path,
        mixtures_dir=mixtures_dir,
        sources_dir=sources_dir,
        output_dir=output_dir,
        folder_names=folder_map,
        rename_sources=rename_sources,
        target_sr=target_sample_rate,
        overwrite=overwrite_existing,
    )
