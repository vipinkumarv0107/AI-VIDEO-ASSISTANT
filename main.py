import shutil
import uuid
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import analyze_transcript
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()


app = FastAPI(
    title="AI Video Assistant",
    description="AI Video Intelligence Backend",
    version="2.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "downloades"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


rag_chains = {}


ALLOWED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".mp3",
    ".wav",
    ".m4a",
    ".webm",
}


def validate_url(url: str) -> str:
    url = url.strip()

    if not url:
        raise HTTPException(
            status_code=400,
            detail="URL cannot be empty."
        )

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(
            status_code=400,
            detail="URL must start with http:// or https://"
        )

    if not parsed.netloc:
        raise HTTPException(
            status_code=400,
            detail="Invalid URL."
        )

    return url


def run_pipeline(
    source: str,
    language: str = "english",
) -> dict:

    print()
    print("=" * 60)
    print("STARTING AI VIDEO ASSISTANT")
    print("=" * 60)

    print("\n[1/5] Processing media...")

    chunks = process_input(source)

    if not chunks:
        raise RuntimeError(
            "No audio chunks were generated."
        )

    print(
        f"Created {len(chunks)} audio chunk(s)."
    )

    print("\n[2/5] Starting transcription...")

    transcript = transcribe_all(
        chunks,
        language=language,
    )

    if not transcript or not transcript.strip():
        raise RuntimeError(
            "Transcription returned empty text."
        )

    print(
        f"Transcript length: {len(transcript)} characters"
    )

    print("Raw transcription preview:")
    print(transcript[:500])

    print("\n[3/5] Running AI analysis...")

    print(
        "Generating title, summary, action items, "
        "decisions and questions using ONE Mistral request..."
    )

    analysis = analyze_transcript(
        transcript
    )

    title = analysis.get(
        "title",
        "Untitled Video"
    )

    summary = analysis.get(
        "summary",
        ""
    )

    action_items = analysis.get(
        "action_items",
        ""
    )

    decisions = analysis.get(
        "key_decisions",
        ""
    )

    questions = analysis.get(
        "open_questions",
        ""
    )

    print("\n[4/5] Building RAG chain...")

    rag_chain = build_rag_chain(
        transcript
    )

    print("\n[5/5] Creating analysis session...")

    analysis_id = str(
        uuid.uuid4()
    )

    rag_chains[analysis_id] = rag_chain

    print()

    print("=" * 60)

    print(
        f"ANALYSIS COMPLETED: {analysis_id}"
    )

    print("=" * 60)

    return {
        "analysis_id": analysis_id,
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
    }


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "AI Video Assistant API is running",
        "version": "2.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/api/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    language: str = Form("english"),
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format: {extension}. "
                f"Allowed formats: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    filename = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    file_path = UPLOAD_DIR / filename

    try:

        print(
            f"Saving uploaded file: {file.filename}"
        )

        with file_path.open("wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

        result = run_pipeline(
            str(file_path),
            language=language,
        )

        return {
            "success": True,
            "source_type": "file",
            "data": result,
        }

    except HTTPException:

        raise

    except Exception as e:

        print(
            f"Analysis error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        await file.close()


@app.post("/api/analyze-url")
async def analyze_url(
    url: str = Form(...),
    language: str = Form("english"),
):

    url = validate_url(url)

    try:

        print()
        print("=" * 60)
        print("ONLINE URL ANALYSIS")
        print("=" * 60)

        print(
            f"URL: {url}"
        )

        result = run_pipeline(
            url,
            language=language,
        )

        result["source_url"] = url

        return {
            "success": True,
            "source_type": "url",
            "data": result,
        }

    except HTTPException:

        raise

    except Exception as e:

        print(
            f"URL analysis error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post("/api/ask")
async def ask_ai(
    analysis_id: str = Form(...),
    question: str = Form(...),
):

    analysis_id = analysis_id.strip()
    question = question.strip()

    if not analysis_id:
        raise HTTPException(
            status_code=400,
            detail="Analysis ID cannot be empty."
        )

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if analysis_id not in rag_chains:
        raise HTTPException(
            status_code=404,
            detail=(
                "Analysis session not found. "
                "Analyze a video first."
            ),
        )

    try:

        rag_chain = rag_chains[
            analysis_id
        ]

        answer = ask_question(
            rag_chain,
            question,
        )

        return {
            "success": True,
            "analysis_id": analysis_id,
            "question": question,
            "answer": answer,
        }

    except Exception as e:

        print(
            f"RAG error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )