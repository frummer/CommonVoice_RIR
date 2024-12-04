import librosa
import torch

import werpy
from transformers import AutoProcessor, Wav2Vec2ForCTC


class MMSArabicASR:
    def __init__(
        self, model_id="facebook/mms-1b-all", target_lang="ara", sampling_rate=16000
    ):
        """
        Initialize the model and processor for Arabic ASR.
        """
        self.model_id = model_id
        self.target_lang = target_lang
        self.sampling_rate = sampling_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Model...")
        self.processor = AutoProcessor.from_pretrained(
            model_id, target_lang=target_lang
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_id, target_lang=target_lang, ignore_mismatched_sizes=True
        ).to(self.device)

    def preprocess_audio(self, waveform, sr):
        """
        Preprocess an audio waveform and resample it if needed.
        """
        if sr != self.sampling_rate:
            print("Resampling audio...")
            waveform = librosa.resample(
                waveform, orig_sr=sr, target_sr=self.sampling_rate
            )
        input_values = self.processor(
            waveform, return_tensors="pt", sampling_rate=self.sampling_rate
        ).input_values.to(self.device)
        return input_values

    def transcribe(self, waveform, sr):
        """
        Perform inference on a preloaded audio waveform and return the transcription.
        """
        input_values = self.preprocess_audio(waveform, sr)
        with torch.no_grad():
            print("Performing inference...")
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription

    @staticmethod
    def calculate_wer(reference, hypothesis):
        """
        Calculate Word Error Rate (WER) between reference and hypothesis transcriptions.
        """
        summary = werpy.wer(reference, hypothesis)
        return summary
    
    @staticmethod
    def normalize(input):
        """
        Calculate Word Error Rate (WER) between reference and hypothesis transcriptions.
        """
        reference = werpy.normalize(input)
        return reference



# Example usage
if __name__ == "__main__":
    audio_path = "C:\\Users\\arifr\\git\\CommonVoice_RIR\\common_voice_ar_25245337.mp3"
    annotated_transcription = "لا تركنن إلى العدو فإنه"

    # Load audio
    print("Loading Wav...")
    waveform, sr = librosa.load(audio_path, sr=None)

    # Initialize the ASR system
    asr_system = MMSArabicASR()

    # Perform transcription
    transcription = asr_system.transcribe(waveform, sr)
    print(f"Transcript: {transcription}")
    print(f"Annotation: {annotated_transcription}")

    # Calculate WER
    # ref = [annotated_transcription]
    # hyp = [transcription]
    # wer_summary = asr_system.calculate_wer(ref, hyp)
    # print(wer_summary)
