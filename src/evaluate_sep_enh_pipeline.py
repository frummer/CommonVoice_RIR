import json
import os

from evaluation_utils.load_files_utils import (
    load_and_validate_config,
    load_audio_and_resmaple,
    load_metadata,
)
from evaluation_utils.mms_arabic_asr import MMSArabicASR


def main():
    """Main function to execute the script."""
    # load config
    config_path = os.getenv(
        "CONFIG_PATH", "./src/evaluate_sep_enh_pipeline_config.json"
    )
    config = load_and_validate_config(file_path=config_path)
    dataset_dir = config["directories"]["dataset_directory"]
    separatd_audios_dir = config["directories"]["separatd_audios_directory"]
    target_sample_rate = config["target_sample_rate"]
    # Step 1: Load metadata
    metadata = load_metadata(dataset_dir=dataset_dir)

    # Step 1.5: Load MMS model.
    # Load the model and processor
    # Initialize the ASR system
    asr_system = MMSArabicASR()
    # Load your audio sample
    # Step 2: Loop over each entry
    for index, entry in enumerate(metadata):
        # Step 3:
        # Step 3.1 load data to ram:
        # load id from entry
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
        # load gt1_transcription
        gt1_transcription = entry["original_files"][0]["transcription"]
        # load gt2_name
        gt2_file_name = entry["original_files"][1]["file"]
        # load gt2_audio_file
        gt2_audio_file_path = os.path.join(dataset_dir, id, gt2_file_name)
        # load and resample
        gt2_audio, _resampled2 = load_audio_and_resmaple(
            audio_path=gt2_audio_file_path, target_sample_rate=target_sample_rate
        )
        # load gt2_transcription
        gt2_transcription = entry["original_files"][1]["transcription"]
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

        # Step 3.2 transcribe all audios:
        # transcribe gt1
        print(f"gt1_transcription:{gt1_transcription}")
        print(f"gt2_transcription:{gt2_transcription}")
        normalized_gt1_transcription = asr_system.normalize([gt1_transcription])
        normalized_gt2_transcription = asr_system.normalize([gt2_transcription])
        print(f"normalized_gt1_transcription:{normalized_gt1_transcription}")
        print(f"normalized_gt2_transcription:{normalized_gt2_transcription}")

        # Perform transcription
        clean_gt_1_transcription = asr_system.transcribe(gt1_audio, 16000)
        # transcribe gt2
        clean_gt_2_transcription = asr_system.transcribe(gt2_audio, 16000)
        print(f"clean_gt_1_transcription:{clean_gt_1_transcription}")
        print(f"clean_gt_2_transcription:{clean_gt_2_transcription}")
        # transcribe audio1_file
        sep_audio_1_transcription = asr_system.transcribe(separated_audio_file_1, 16000)
        sep_audio_2_transcription = asr_system.transcribe(separated_audio_file_2, 16000)
        print(f"sep_audio_1_transcription:{sep_audio_1_transcription}")
        print(f"sep_audio_2_transcription:{sep_audio_2_transcription}")


        # Step 3.3 calculate WER:
        # transcribe gt1 vs transcription_g1
        # transcribe gt2 vs transcription_g2
        clean1_wer_summary = asr_system.calculate_wer([normalized_gt1_transcription], [clean_gt_1_transcription])
        clean2_wer_summary = asr_system.calculate_wer([normalized_gt2_transcription], [clean_gt_2_transcription])
        # transcribe gt1_audio_file vs transcription_g1
        # transcribe gt1_audio_file vs transcription_g2
        separated1_1_wer_summary = asr_system.calculate_wer([normalized_gt1_transcription], [sep_audio_1_transcription])
        separated1_2_wer_summary = asr_system.calculate_wer([normalized_gt1_transcription], [sep_audio_2_transcription])
        # transcribe gt2_audio_file vs transcription_g1
        # transcribe gt2_audio_file vs transcription_g2
        separated2_1_wer_summary = asr_system.calculate_wer([normalized_gt2_transcription], [sep_audio_1_transcription])
        separated2_2_wer_summary = asr_system.calculate_wer([normalized_gt2_transcription], [sep_audio_2_transcription])
        print(f"clean1_wer_summary:{clean1_wer_summary}")
        print(f"clean2_wer_summary:{clean2_wer_summary}")
        print(f"separated1_1_wer_summary:{separated1_1_wer_summary}")
        print(f"separated1_2_wer_summary:{separated1_2_wer_summary}")
        print(f"separated2_1_wer_summary:{separated2_1_wer_summary}")
        print(f"separated2_2_wer_summary:{separated2_2_wer_summary}")

        print(f"index:{index}")
        break


if __name__ == "__main__":
    main()
