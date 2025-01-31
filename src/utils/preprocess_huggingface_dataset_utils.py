import time

from datasets import Dataset


def preprocess_dataset(
    dataset,
    long_length_overlapped_samples_amount: int,
    short_length_overlapped_samples_amount: int,
    mixed_length_overlapped_samples_amount: int,
    long_audio_threshold: int,
    short_audio_threshold: int,
):
    if mixed_length_overlapped_samples_amount % 2 != 0:
        mixed_length_overlapped_samples_amount += 1
    start_time = time.time()
    min_length_dataset = dataset.filter(
        filter_long_audio, fn_kwargs={"min_duration": long_audio_threshold}
    )
    min_length_dataset_length = len(min_length_dataset)
    subset_min_length_dataset = min_length_dataset.select(
        range(
            2 * long_length_overlapped_samples_amount
            + mixed_length_overlapped_samples_amount
        )
    )
    range_dataset = dataset.filter(
        filter_duration_range,
        fn_kwargs={
            "min_duration": short_audio_threshold,
            "max_duration": long_audio_threshold,
        },
    )
    subset_range_length_dataset = range_dataset.select(
        range(
            2 * short_length_overlapped_samples_amount
            + mixed_length_overlapped_samples_amount
        )
    )
    # ----------------------
    # Split Each Subset into First Half + Remainder
    # ----------------------

    lenA = len(subset_min_length_dataset)
    lenB = len(subset_range_length_dataset)

    halfA = lenA // 2
    halfB = lenB // 2

    subsetA_half = subset_min_length_dataset.select(range(halfA))
    subsetA_remainder = subset_min_length_dataset.select(range(halfA, lenA))

    subsetB_half = subset_range_length_dataset.select(range(halfB))
    subsetB_remainder = subset_range_length_dataset.select(range(halfB, lenB))

    print("Subset A half:", len(subsetA_half), " | remainder:", len(subsetA_remainder))
    print("Subset B half:", len(subsetB_half), " | remainder:", len(subsetB_remainder))

    # ----------------------
    # 4) Build Final Dataset in the Required Order
    # ----------------------
    #
    # Order:
    #   (1) All from 'subsetA_half',
    #   (2) All from 'subsetB_half',
    #   (3) Interleave the remainders: [A_rem[0], B_rem[0], A_rem[1], B_rem[1], ...],
    #       plus any leftover if one remainder is larger than the other.

    final_list = []

    # Part 1: Append the first half of A
    for i in range(len(subsetA_half)):
        final_list.append(subsetA_half[i])

    # Part 2: Append the first half of B
    for i in range(len(subsetB_half)):
        final_list.append(subsetB_half[i])

    # Part 3: Interleave the remainders
    n = min(len(subsetA_remainder), len(subsetB_remainder))
    for i in range(n):
        final_list.append(subsetA_remainder[i])
        final_list.append(subsetB_remainder[i])

    # Convert the final list of dicts to a Hugging Face Dataset
    final_dataset = Dataset.from_list(final_list)

    print(
        f"Finished pre-processings data. took: {round(time.time()-start_time,2)} seconds"
    )

    return final_dataset


def filter_duration_range(example, min_duration=5.0, max_duration=10.0):
    audio_array = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]
    duration_sec = len(audio_array) / sr
    return (duration_sec >= min_duration) and (duration_sec <= max_duration)


def filter_long_audio(example, min_duration=10.0):
    """Filters audio files longer than min_duration seconds."""
    duration = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
    return duration > min_duration
