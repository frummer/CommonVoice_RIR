#!/usr/bin/env python3

import os
from typing import List, Tuple

from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_AUTH_TOKEN = "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"
"""
Hugging Face Auth Token.
Change this to your actual token, or handle it via environment variables if needed.
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def segments_match(seg1: Segment, seg2: Segment, tol: float = 0.05) -> bool:
    """
    Check if two pyannote.core.Segment objects match within a time tolerance.

    :param seg1: Segment object (start, end)
    :param seg2: Segment object (start, end)
    :param tol:  Time tolerance (seconds). Defaults to 0.05 (50 ms).
    :return:     True if starts OR ends are within tol, False otherwise.
                 (Note the 'or' in this condition—adjust if needed.)
    """
    return abs(seg1.start - seg2.start) <= tol and abs(seg1.end - seg2.end) <= tol


def get_vad_segments(file_path: str, pipeline: Pipeline) -> list:
    """
    Run VAD on a single audio file and return a list of pyannote.core.Segment objects.

    :param file_path: Path to the audio file
    :param pipeline:  Preloaded pyannote.audio Pipeline for VAD
    :return:          List of Segment objects
    """
    annotation: Annotation = pipeline(file_path)
    return list(annotation.itersegments())


def count_matched_segments(segments_source, segments_sep, tol: float) -> int:
    """
    Count how many segments match (within a tolerance) between a list of source
    segments and a list of separated segments.

    :param segments_source: List of Segment objects (source)
    :param segments_sep:    List of Segment objects (separated)
    :param tol:             Time tolerance (seconds)
    :return:                Number of matched segments
    """
    matched_pairs = 0
    for src_seg in segments_source:
        for sep_seg in segments_sep:
            if segments_match(src_seg, sep_seg, tol):
                matched_pairs += 1
    return matched_pairs


def run_vad_comparison(
    SOURCE1_WAV: str,
    SOURCE2_WAV: str,
    SEP1_WAV: str,
    SEP2_WAV: str,
    TOLERANCE: float,
    verbose: bool = True,
) -> List[Tuple[str, str]]:
    """
    Performs VAD on two source audio files and two separated audio files, and
    determines which separated file is the best match for each source based
    on the number of matched segments.

    :param SOURCE1_WAV: Path to source 1 audio file
    :param SOURCE2_WAV: Path to source 2 audio file
    :param SEP1_WAV:    Path to separated audio 1
    :param SEP2_WAV:    Path to separated audio 2
    :param TOLERANCE:   Float (seconds) for time tolerance in matching segments
    :param verbose:     Whether to print detailed information. Defaults to True.
    :return:            A list of two tuples:
                        [
                            (SOURCE1_WAV, best_matched_separated_path_for_source1),
                            (SOURCE2_WAV, best_matched_separated_path_for_source2)
                        ]
    """

    # 1. Load the Pyannote VAD pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection", use_auth_token=HF_AUTH_TOKEN
    )

    # 2. Run VAD on each audio file
    source1_segments = get_vad_segments(SOURCE1_WAV, pipeline)
    source2_segments = get_vad_segments(SOURCE2_WAV, pipeline)
    sep1_segments = get_vad_segments(SEP1_WAV, pipeline)
    sep2_segments = get_vad_segments(SEP2_WAV, pipeline)

    # 3. Compare all possible source-sep pairings
    matching_s1_sep1 = count_matched_segments(
        source1_segments, sep1_segments, TOLERANCE
    )
    matching_s1_sep2 = count_matched_segments(
        source1_segments, sep2_segments, TOLERANCE
    )
    matching_s2_sep1 = count_matched_segments(
        source2_segments, sep1_segments, TOLERANCE
    )
    matching_s2_sep2 = count_matched_segments(
        source2_segments, sep2_segments, TOLERANCE
    )

    # 4. Optionally print out the results for each pairing
    if verbose:
        print("\n===== MATCH COUNTS FOR EACH PAIRING =====")
        print(
            f"{os.path.basename(SOURCE1_WAV)} vs. {os.path.basename(SEP1_WAV)}: {matching_s1_sep1}"
        )
        print(
            f"{os.path.basename(SOURCE1_WAV)} vs. {os.path.basename(SEP2_WAV)}: {matching_s1_sep2}"
        )
        print(
            f"{os.path.basename(SOURCE2_WAV)} vs. {os.path.basename(SEP1_WAV)}: {matching_s2_sep1}"
        )
        print(
            f"{os.path.basename(SOURCE2_WAV)} vs. {os.path.basename(SEP2_WAV)}: {matching_s2_sep2}"
        )

    # 5. Decide which separated file is best for each source
    if matching_s1_sep1 >= matching_s1_sep2:
        best_for_source1 = (SEP1_WAV, matching_s1_sep1)
    else:
        best_for_source1 = (SEP2_WAV, matching_s1_sep2)

    if matching_s2_sep1 >= matching_s2_sep2:
        best_for_source2 = (SEP1_WAV, matching_s2_sep1)
    else:
        best_for_source2 = (SEP2_WAV, matching_s2_sep2)

    # Optionally print final “best match” decision
    if verbose:
        print("\n===== BEST MATCH DECISIONS =====")
        print(
            f"Best match for {os.path.basename(SOURCE1_WAV)} is {os.path.basename(best_for_source1[0])} "
            f"with {best_for_source1[1]} matched segments."
        )
        print(
            f"Best match for {os.path.basename(SOURCE2_WAV)} is {os.path.basename(best_for_source2[0])} "
            f"with {best_for_source2[1]} matched segments."
        )

    # 6. Return a list of two tuples:
    #    (source audio path, best matched separated audio path)
    return [
        (SOURCE1_WAV, best_for_source1[0]),
        (SOURCE2_WAV, best_for_source2[0]),
    ]


def main() -> List[Tuple[str, str]]:
    """
    Example usage of run_vad_comparison function.
    Returns a list of two tuples containing the source path and the matched extracted audio path.
    """

    # Example file paths (replace these with your actual paths):
    SOURCE2_WAV = (
        "C:\\Users\\arifr\\git\\CommonVoice_RIR\\sources_samples\\0_padded.flac"
    )
    SOURCE1_WAV = (
        "C:\\Users\\arifr\\git\\CommonVoice_RIR\\sources_samples\\1_padded.flac"
    )
    SEP1_WAV = (
        "C:\\Users\\arifr\\git\\CommonVoice_RIR\\separated_samples\\"
        "00a4da90-728a-41a1-b4d4-2a18ddaf80c6_spk1_corrected.wav"
    )
    SEP2_WAV = (
        "C:\\Users\\arifr\\git\\CommonVoice_RIR\\separated_samples\\"
        "00a4da90-728a-41a1-b4d4-2a18ddaf80c6_spk2_corrected.wav"
    )

    # Example tolerance
    TOLERANCE = 0.35

    # Run the comparison, controlling verbosity as desired:
    # Set verbose=True to print details, verbose=False to suppress prints
    result = run_vad_comparison(
        SOURCE1_WAV, SOURCE2_WAV, SEP1_WAV, SEP2_WAV, TOLERANCE, verbose=True
    )

    return result


if __name__ == "__main__":
    # When running as a script, we'll capture the result from main:
    final_pairs = main()
    print("\nReturned from main():", final_pairs)
