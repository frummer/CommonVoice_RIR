# iterate over metadata.json file

# load relevant files

# calculate metrics

import os
from os.path import join

import pandas as pd
from pesq import pesq

from evaluation_utils.additional_metrics_utils import (
    energy_ratios,
    print_mean_std,
    stoi,
)
from evaluation_utils.load_files_utils import (
    load_and_validate_config,
    load_audio_and_resmaple,
    load_metadata,
)


def main():
    """Main function to execute the script."""
    # load config
    config_path = os.getenv(
        "CONFIG_PATH", "./src/evaluate_metrics_sep_enh_pipeline_config.json"
    )
    config = load_and_validate_config(file_path=config_path)
    dataset_dir = config["directories"]["dataset_directory"]
    separatd_audios_dir = config["directories"]["separatd_audios_directory"]
    target_sample_rate = config["target_sample_rate"]
    # Step 1: Load metadata
    metadata = load_metadata(dataset_dir=dataset_dir)
    data = {
        "filename": [],
        "pesq": [],
        "estoi": [],
        "si_sdr": [],
        "si_sir": [],
        "si_sar": [],
    }
    # Step 2: Loop over each entry
    for index, entry in enumerate(metadata):
        (
            gt1_file_name,
            gt1_audio,
            gt2_file_name,
            gt2_audio,
            separated_audio_file_1,
            separated_audio_file_2,
        ) = load_wavs_from_entry(
            dataset_dir, separatd_audios_dir, target_sample_rate, entry
        )
        print(f"len(separated_audio_file_2):{len(separated_audio_file_2)}")
        print(f"len(separated_audio_file_1):{len(separated_audio_file_1)}")
        print(f"len(gt2_audio):{len(gt2_audio)}")
        print(f"len(gt1_audio):{len(gt1_audio)}")

        sep_file1_1 = separated_audio_file_2[..., : len(gt2_audio)]
        sep_file1_2 = separated_audio_file_1[..., : len(gt1_audio)]
        n1_1 = sep_file1_1 - gt2_audio
        n1_2 = sep_file1_2 - gt1_audio

        pesq_1_1, estoi_1_1, si_sdr_1_1, si_sir_1_1, si_sar_1_1 = get_metrics(
            sr=target_sample_rate, x=gt2_audio, x_hat=sep_file1_1, noise=n1_1
        )
        pesq_1_2, estoi_1_2, si_sdr_1_2, si_sir_1_2, si_sar_1_2 = get_metrics(
            sr=target_sample_rate, x=gt1_audio, x_hat=sep_file1_2, noise=n1_2
        )

        sep_file2_1 = separated_audio_file_2[..., : len(gt1_audio)]
        sep_file2_2 = separated_audio_file_1[..., : len(gt2_audio)]
        n2_1 = sep_file2_1 - gt1_audio
        n2_2 = sep_file2_2 - gt2_audio

        pesq_2_1, estoi_2_1, si_sdr_2_1, si_sir_2_1, si_sar_2_1 = get_metrics(
            sr=target_sample_rate, x=gt1_audio, x_hat=sep_file2_1, noise=n2_1
        )
        pesq_2_2, estoi_2_2, si_sdr_2_2, si_sir_2_2, si_sar_2_2 = get_metrics(
            sr=target_sample_rate, x=gt2_audio, x_hat=sep_file2_2, noise=n2_2
        )
        if (estoi_2_1 > estoi_1_1 and estoi_2_2 > estoi_1_2) and (
            pesq_2_1 > pesq_1_1 and pesq_2_2 > pesq_1_2
        ):
            # pesq_2_1, estoi_2_1, si_sdr_2_1, si_sir_2_1, si_sar_2_1
            data["filename"].append(gt1_file_name)
            data["pesq"].append(pesq_2_1)
            data["estoi"].append(estoi_2_1)
            data["si_sdr"].append(si_sdr_2_1)
            data["si_sir"].append(si_sir_2_1)
            data["si_sar"].append(si_sar_2_1)

            data["filename"].append(gt2_file_name)
            data["pesq"].append(pesq_2_2)
            data["estoi"].append(estoi_2_2)
            data["si_sdr"].append(si_sdr_2_2)
            data["si_sir"].append(si_sir_2_2)
            data["si_sar"].append(si_sar_2_2)

        elif (estoi_2_1 < estoi_1_1 and estoi_2_2 < estoi_1_2) and (
            pesq_2_1 < pesq_1_1 and pesq_2_2 < pesq_1_2
        ):
            data["filename"].append(gt2_file_name)
            data["pesq"].append(pesq_1_1)
            data["estoi"].append(estoi_1_1)
            data["si_sdr"].append(si_sdr_1_1)
            data["si_sir"].append(si_sir_1_1)
            data["si_sar"].append(si_sar_1_1)

            data["filename"].append(gt1_file_name)
            data["pesq"].append(pesq_1_2)
            data["estoi"].append(estoi_1_2)
            data["si_sdr"].append(si_sdr_1_2)
            data["si_sir"].append(si_sir_1_2)
            data["si_sar"].append(si_sar_1_2)
        # Save results as DataFrame
    df = pd.DataFrame(data)
    df.to_csv(join(separatd_audios_dir, "_results.csv"), index=False)

    # Save average results
    text_file = join(separatd_audios_dir, "_avg_results.txt")
    with open(text_file, "w") as file:
        file.write("PESQ: {} \n".format(print_mean_std(data["pesq"])))
        file.write("ESTOI: {} \n".format(print_mean_std(data["estoi"])))
        file.write("SI-SDR: {} \n".format(print_mean_std(data["si_sdr"])))
        file.write("SI-SIR: {} \n".format(print_mean_std(data["si_sir"])))
        file.write("SI-SAR: {} \n".format(print_mean_std(data["si_sar"])))


def get_metrics(sr, x, x_hat, noise):
    try:
        p = pesq(sr, x, x_hat, "nb")
    except:
        p = float("nan")
    pesq_result = p
    estoi = stoi(x, x_hat, sr, extended=True)
    energy_ratios_results = energy_ratios(x_hat, x, noise)
    si_sdr = energy_ratios_results[0]
    si_sir = energy_ratios_results[1]
    si_sar = energy_ratios_results[2]
    return pesq_result, estoi, si_sdr, si_sir, si_sar


def load_wavs_from_entry(dataset_dir, separatd_audios_dir, target_sample_rate, entry):
    id = entry["id"]
    # load - reference and transcriptions
    # load gt1_name
    gt1_file_name = entry["original_files"][0]["file"]
    # load gt1_audio_file
    gt1_audio_file_path = os.path.join(dataset_dir, id, gt1_file_name)
    # load and resample
    gt1_audio, _resampled1 = load_audio_and_resmaple(
        audio_path=gt1_audio_file_path, target_sample_rate=target_sample_rate
    )
    # load gt2_name
    gt2_file_name = entry["original_files"][1]["file"]
    # load gt2_audio_file
    gt2_audio_file_path = os.path.join(dataset_dir, id, gt2_file_name)
    # load and resample
    gt2_audio, _resampled2 = load_audio_and_resmaple(
        audio_path=gt2_audio_file_path, target_sample_rate=target_sample_rate
    )
    # load - separated audio1
    separated_file_1_path = os.path.join(
        separatd_audios_dir, f"ff_{id}_noisy_with_music_opus_spk1_corrected.wav"
    )
    separated_audio_file_1, _resampled3 = load_audio_and_resmaple(
        audio_path=separated_file_1_path, target_sample_rate=target_sample_rate
    )
    # load - separated audio2
    separated_file_2_path = os.path.join(
        separatd_audios_dir, f"ff_{id}_noisy_with_music_opus_spk2_corrected.wav"
    )
    separated_audio_file_2, _resampled4 = load_audio_and_resmaple(
        audio_path=separated_file_2_path, target_sample_rate=target_sample_rate
    )

    return (
        gt1_file_name,
        gt1_audio,
        gt2_file_name,
        gt2_audio,
        separated_audio_file_1,
        separated_audio_file_2,
    )


if __name__ == "__main__":
    main()
