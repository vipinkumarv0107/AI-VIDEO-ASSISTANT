import os
import uuid
from pathlib import Path

from pydub import AudioSegment


DOWNLOAD_DIR = Path("downloades")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Optional: automatically add a common Windows FFmpeg location
# if it exists. Otherwise FFmpeg must already be on PATH.
COMMON_FFMPEG_BIN = Path(
    os.environ.get(
        "LOCALAPPDATA",
        ""
    )
) / (
    "Microsoft"
    "/WinGet/Packages/Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
    "/ffmpeg-9.0.1-full_build-shared/bin"
)

if COMMON_FFMPEG_BIN.exists():
    os.environ["PATH"] = (
        str(COMMON_FFMPEG_BIN)
        + os.pathsep
        + os.environ.get("PATH", "")
    )


def download_online_media(url: str) -> str:
    """
    Download audio from YouTube and other websites supported by yt-dlp.
    """

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is not installed. Run: pip install yt-dlp"
        ) from exc

    output_id = uuid.uuid4().hex

    output_template = str(
        DOWNLOAD_DIR / f"{output_id}.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    print("Downloading online media...")
    print(f"URL: {url}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(
                url,
                download=True,
            )

    except Exception as exc:
        raise RuntimeError(
            f"Could not download the URL. "
            f"Make sure the URL is public and supported by yt-dlp. "
            f"Details: {exc}"
        ) from exc

    wav_path = DOWNLOAD_DIR / f"{output_id}.wav"

    if wav_path.exists():
        return str(wav_path)

    candidates = list(
        DOWNLOAD_DIR.glob(f"{output_id}.*")
    )

    if not candidates:
        raise RuntimeError(
            "Download finished but the audio file was not found."
        )

    # If post-processing did not produce WAV, convert the result.
    return convert_to_wav(str(candidates[0]))


def convert_to_wav(input_path: str) -> str:
    """Convert audio/video to mono 16 kHz WAV."""

    input_path = str(input_path)

    output_path = (
        Path(input_path).with_suffix("")
        .with_name(
            Path(input_path).stem + "_converted.wav"
        )
    )

    print(f"Converting to WAV: {input_path}")

    try:
        audio = AudioSegment.from_file(input_path)
    except Exception as exc:
        raise RuntimeError(
            "FFmpeg could not read this media file. "
            "Check that FFmpeg is installed and available on PATH."
        ) from exc

    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
    )

    audio.export(
        output_path,
        format="wav",
    )

    return str(output_path)


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10,
) -> list:
    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start in enumerate(
        range(0, len(audio), chunk_ms)
    ):
        chunk = audio[start:start + chunk_ms]

        chunk_path = (
            f"{wav_path}_chunk_{i}.wav"
        )

        chunk.export(
            chunk_path,
            format="wav",
        )

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    source = str(source).strip()

    if not source:
        raise ValueError("Input source is empty.")

    if source.startswith(("http://", "https://")):
        print("Detected online URL.")
        wav_path = download_online_media(source)
    else:
        print("Detected local file.")

        if not os.path.exists(source):
            raise FileNotFoundError(
                f"File not found: {source}"
            )

        wav_path = convert_to_wav(source)

    print("Chunking audio...")

    chunks = chunk_audio(wav_path)

    if not chunks:
        raise RuntimeError(
            "No audio chunks were created."
        )

    print(
        f"Audio ready - {len(chunks)} chunk(s) created."
    )

    return chunks


if __name__ == "__main__":
    test_url = (
        "https://youtu.be/s2HYl12gmOY"
    )

    data = process_input(test_url)
    print(data)
