import os
import tempfile
from pathlib import Path
import subprocess
import librosa
import torch

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

BASE_DIR = Path(__file__).resolve().parents[2]


FFMPEG_EXE = "ffmpeg"

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "whisper-small-torgo"
)


class WhisperTranscriber:

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Loading Whisper on {self.device}..."
        )

        self.processor = (
            WhisperProcessor.from_pretrained(
                MODEL_DIR,
                local_files_only=True,
            )
        )

        self.model = (
            WhisperForConditionalGeneration
            .from_pretrained(
                MODEL_DIR,
                local_files_only=True,
            )
        )

        self.model.to(self.device)
        self.model.eval()

        self.model.generation_config.language = "en"
        self.model.generation_config.task = "transcribe"

        print("Whisper model loaded.")

    def transcribe(
            self,
            audio_path: str,
    ) -> str:

        converted_path = None

        try:

            # Convert uploaded audio to:
            # WAV / 16 kHz / mono

            converted_path = self.convert_to_wav(
                audio_path
            )

            waveform, sampling_rate = librosa.load(
                converted_path,
                sr=16000,
                mono=True,
            )

            inputs = self.processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
            )

            input_features = (
                inputs.input_features.to(self.device)
            )

            with torch.inference_mode():

                generated_ids = self.model.generate(
                    input_features,
                    language="en",
                    task="transcribe",
                    max_new_tokens=225,
                    num_beams=1,
                )

            transcription = (
                self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]
                .strip()
            )

            #Returning the predicted text
            return transcription

        finally:

            if (
                    converted_path
                    and os.path.exists(converted_path)
            ):
                os.remove(converted_path)

    def convert_to_wav(self, audio_path: str) -> str:
        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        )

        output_path = output_file.name

        output_file.close()

        subprocess.run(
            [
                FFMPEG_EXE,
                "-y",
                "-i",
                audio_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        return output_path
