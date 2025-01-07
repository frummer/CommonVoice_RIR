import argparse
import json
import os

import jiwer
from datasets import Audio, load_dataset

from align_trancsription_to_separated_wav import run_vad_comparison
from evaluation_utils.mms_arabic_asr import MMSASR  # Import the class


def get_text_for_audio_source(supervisions, audio_path, asr_system):
    """
    Aligns text to the audio source by running the MMS model and calculating CER.

    Args:
        supervisions (list): A list of supervision dicts.
                             e.g., [{'texts': ['some text'], ...}, ...]
        audio_path (str): Path to the audio file to be transcribed.
        asr_system (MMSASR): Instance of the ASR system to transcribe audio.

    Returns:
        str: The best matched text for the given audio source.
    """
    # 1) Run the speech recognition model on the audio
    transcription = asr_system.transcribe(audio_path, 16000)

    # 2) Use CER to align and choose the best match from supervisions
    if supervisions:
        best_match = min(
            supervisions,
            key=lambda x: jiwer.cer(
                x["texts"][0], transcription
            ),  # compare to first text variant
        )
        return best_match["texts"][0]

    # If no supervisions exist, just return the raw transcription
    return transcription


def parse_mixedcut_line(
    json_line: str,
    source1_base_path: str,
    source2_base_path: str,
    separated_audios_base_path: str,
    asr_system: MMSASR,
):
    """
    Parse the raw JSON line, aligning audio paths and supervisions,
    and returning a list of dataset entries (two per 'tracks' entry):
      - one for 'source1'
      - one for 'source2'

    Each entry dict:
      {
          "id": f"{mixedcut_id}_source1",
          "mixedcut_id": mixedcut_id,
          "audio_path": "/path/to/source1.wav",
          "text": "...",
          "start_sec": float,
          "duration_sec": float
      }
    """
    item = json.loads(json_line)
    mixedcut_id = item["id"]

    results = []

    # We iterate over 'tracks' to produce entries.
    # Each track might correspond to a segment or partial mix, but we'll produce *two* entries:
    #   one for source1, one for source2
    for track in item.get("tracks", []):
        cut = track.get("cut", {})
        supervisions = cut.get("supervisions", [])

        # Build the file paths for the 'original' source audios
        source1_path = os.path.join(source1_base_path, f"{mixedcut_id}.flac")
        source2_path = os.path.join(source2_base_path, f"{mixedcut_id}.flac")

        # Build the file paths for the separated audios (using naming convention <mixedcut_id>_sep1.wav, etc.)
        sep1_path = os.path.join(separated_audios_base_path, f"{mixedcut_id}_sep1.wav")
        sep2_path = os.path.join(separated_audios_base_path, f"{mixedcut_id}_sep2.wav")

        # Run VAD-based comparison between original and separated audios (example usage)
        aligned_sources = run_vad_comparison(
            source1_path, source2_path, sep1_path, sep2_path, threshold=0.5
        )

        # aligned_sources is assumed to be something like:
        #   [(0, "/path/to/aligned_source1.wav"), (1, "/path/to/aligned_source2.wav")]
        # so we retrieve the second element from each tuple as the actual path
        aligned_source1_path = aligned_sources[0][1]
        aligned_source2_path = aligned_sources[1][1]

        # Get the text for each separated source
        text1 = get_text_for_audio_source(
            supervisions, aligned_source1_path, asr_system
        )
        text2 = get_text_for_audio_source(
            supervisions, aligned_source2_path, asr_system
        )

        # Add to results: one row for source1, one for source2
        results.append(
            {
                "id": f"{mixedcut_id}_source1",
                "mixedcut_id": mixedcut_id,
                "audio_path": aligned_source1_path,
                "text": text1,
                "start_sec": cut.get("start", 0.0),
                "duration_sec": cut.get("duration", 0.0),
            }
        )
        results.append(
            {
                "id": f"{mixedcut_id}_source2",
                "mixedcut_id": mixedcut_id,
                "audio_path": aligned_source2_path,
                "text": text2,
                "start_sec": cut.get("start", 0.0),
                "duration_sec": cut.get("duration", 0.0),
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Parse MixedCut JSONL into per-audio-source JSONL."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the raw MixedCut JSONL file."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the simpler JSONL file.",
    )
    parser.add_argument(
        "--source1_base_path",
        type=str,
        required=True,
        help="Base path for source 1 audio files (the original audio).",
    )
    parser.add_argument(
        "--source2_base_path",
        type=str,
        required=True,
        help="Base path for source 2 audio files (the original audio).",
    )
    parser.add_argument(
        "--separated_audios_base_path",
        type=str,
        required=True,
        help="Base path for the separated audio files (e.g. <mixedcut_id>_sep1.wav).",
    )
    parser.add_argument(
        "--load_dataset",
        action="store_true",
        help="If set, load the resulting JSONL as a Hugging Face dataset and print a sample.",
    )
    args = parser.parse_args()

    # Initialize the ASR system (for Arabic, English, etc. - here we used 'eng' as an example)
    asr_system = MMSASR(target_lang="eng")

    # Read and parse the JSONL file, then create simplified JSONL
    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            entries = parse_mixedcut_line(
                json_line=line,
                source1_base_path=args.source1_base_path,
                source2_base_path=args.source2_base_path,
                separated_audios_base_path=args.separated_audios_base_path,
                asr_system=asr_system,
            )
            for e in entries:
                fout.write(json.dumps(e) + "\n")

    # (Optional) Load this new JSONL into a Hugging Face Dataset
    if args.load_dataset:
        dataset = load_dataset("json", data_files=args.output, split="train")

        # Cast the audio_path to an Audio feature (sampling rate is just an example)
        dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16_000))

        # Inspect the first sample
        print("Dataset loaded! Number of samples:", len(dataset))
        print("First sample:", dataset[0])


if __name__ == "__main__":
    main()
