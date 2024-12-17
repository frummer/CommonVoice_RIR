import os

from pyannote.audio import Pipeline

# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
# or for VAD specifically:
pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token="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",
)

paths = [
    "C:\\Users\\arifr\\git\\CommonVoice_RIR\\dataset_and_output_files\\overlapped_samples\\02_12_2024_23_17_09_9_-9_12_-3_-3_-9\\4bf1f19b-bbfe-4b51-909e-3e2c8420e631\\common_voice_ar_22770267.mp3",
    "C:\\Users\\arifr\\git\\CommonVoice_RIR\\dataset_and_output_files\\overlapped_samples\\02_12_2024_23_17_09_9_-9_12_-3_-3_-9\\4bf1f19b-bbfe-4b51-909e-3e2c8420e631\\common_voice_ar_24206041.mp3",
    "C:\\Users\\arifr\\git\\CommonVoice_RIR\\dataset_and_output_files\\overlapped_samples\\02_12_2024_23_17_09_9_-9_12_-3_-3_-9\\4bf1f19b-bbfe-4b51-909e-3e2c8420e631\\ff_4bf1f19b-bbfe-4b51-909e-3e2c8420e631_noisy_with_music.wav",
    "C:\\Users\\arifr\\git\\CommonVoice_RIR\\dataset_and_output_files\\separated_samples\\ff_4bf1f19b-bbfe-4b51-909e-3e2c8420e631_noisy_with_music_opus_spk1_corrected.wav",
    "C:\\Users\\arifr\\git\\CommonVoice_RIR\\dataset_and_output_files\\separated_samples\\ff_4bf1f19b-bbfe-4b51-909e-3e2c8420e631_noisy_with_music_opus_spk2_corrected.wav",
]
for path in paths:
    vad_output = pipeline(path)

    # .support() merges overlapping segments into continuous intervals
    merged_speech_timeline = vad_output.get_timeline().support()
    for seg in merged_speech_timeline:
        file_name = os.path.basename(path)
        print(f"file name:{file_name}, Speech from {seg.start:.3f}s to {seg.end:.3f}s")
