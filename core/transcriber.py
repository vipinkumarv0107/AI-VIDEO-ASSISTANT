import os

import requests
import whisper
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

SARVAM_PIECE_SECONDS = 25

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "small",
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
)

SARVAM_API_KEY = os.getenv(
    "SARVAM_API_KEY"
)

SARVAM_STT_TRANSLATE_URL = (
    "https://api.sarvam.ai/speech-to-text-translate"
)

SARVAM_MODEL = os.getenv(
    "SARVAM_STT_MODEL",
    "saaras:v2.5",
)

_model = None


def load_model():
    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}")

        _model = whisper.load_model(
            WHISPER_MODEL,
            download_root=MODEL_DIR,
        )

        print("Whisper model loaded successfully.")

    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()

    result = model.transcribe(
        chunk_path,
        task="transcribe",
        fp16=False,
    )

    return result.get("text", "").strip()


def _send_to_sarvam(piece_path: str) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in your .env file."
        )

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
    }

    with open(piece_path, "rb") as audio_file:
        files = {
            "file": (
                os.path.basename(piece_path),
                audio_file,
                "audio/wav",
            )
        }

        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false",
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(
            f"Sarvam returned {response.status_code}: "
            f"{response.text}"
        )
        response.raise_for_status()

    result = response.json()

    return result.get("transcript", "").strip()


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in your .env file."
        )

    audio = AudioSegment.from_wav(chunk_path)

    piece_ms = SARVAM_PIECE_SECONDS * 1000
    full_text = []

    for i, start in enumerate(
        range(0, len(audio), piece_ms)
    ):
        piece = audio[start:start + piece_ms]

        piece_path = f"{chunk_path}_sv_{i}.wav"

        piece.export(
            piece_path,
            format="wav",
        )

        try:
            print(
                f"  -> Sarvam piece {i + 1}..."
            )

            text = _send_to_sarvam(piece_path)

            if text:
                full_text.append(text)

        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(full_text).strip()


def transcribe_chunk(
    chunk_path: str,
    language: str = "english",
) -> str:
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)

    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(
    chunks: list,
    language: str = "english",
) -> str:
    if not chunks:
        raise ValueError("No audio chunks supplied.")

    full_transcript = []

    engine = (
        "Sarvam AI"
        if language.lower() == "hinglish"
        else "Whisper"
    )

    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(
            f"Transcribing chunk {i + 1}/{len(chunks)}..."
        )

        text = transcribe_chunk(
            chunk,
            language=language,
        )

        if text:
            full_transcript.append(text)

    print("Transcription complete.")

    return " ".join(full_transcript).strip()
