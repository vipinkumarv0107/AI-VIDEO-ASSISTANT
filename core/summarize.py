import os
import time

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

MAX_ANALYSIS_CHARS = 60000


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is missing from your .env file."
        )

    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        mistral_api_key=api_key,
        temperature=0.2,
        max_retries=0,
    )


def analyze_transcript(transcript: str) -> dict:
    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    # Prevent an unexpectedly huge request from exceeding the model context.
    transcript_for_analysis = transcript[:MAX_ANALYSIS_CHARS]

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert meeting and video transcript analyst.

Analyze the transcript and return EXACTLY these section headings:

TITLE:
A short professional title, maximum 8 words.

SUMMARY:
A concise professional summary using bullet points.

ACTION_ITEMS:
A numbered list. For every item include:
- Task
- Owner
- Deadline

If no action items exist, write:
No action items found.

KEY_DECISIONS:
A numbered list of important decisions.

If no decisions exist, write:
No key decisions found.

OPEN_QUESTIONS:
A numbered list of unresolved questions or follow-up topics.

If no open questions exist, write:
No open questions found.

Do not invent facts, owners, deadlines, decisions, or questions.
Use only information supported by the transcript.
Do not add any other section headings.
""",
        ),
        ("human", "{transcript}"),
    ])

    chain = prompt | llm | StrOutputParser()

    last_error = None

    for attempt in range(3):
        try:
            result = chain.invoke({
                "transcript": transcript_for_analysis
            })
            return parse_analysis(result)

        except Exception as exc:
            last_error = exc
            error_text = str(exc)

            if "429" not in error_text and "rate_limit" not in error_text.lower():
                raise

            wait_time = 10 * (attempt + 1)
            print(
                f"Mistral rate limit reached. "
                f"Waiting {wait_time} seconds..."
            )
            time.sleep(wait_time)

    raise RuntimeError(
        f"Mistral API rate limit exceeded after 3 attempts: {last_error}"
    )


def parse_analysis(text: str) -> dict:
    result = {
        "title": "",
        "summary": "",
        "action_items": "",
        "key_decisions": "",
        "open_questions": "",
    }

    sections = {
        "TITLE:": "title",
        "SUMMARY:": "summary",
        "ACTION_ITEMS:": "action_items",
        "KEY_DECISIONS:": "key_decisions",
        "OPEN_QUESTIONS:": "open_questions",
    }

    current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        upper = line.upper()

        if upper in sections:
            current = sections[upper]
            continue

        if current:
            if result[current]:
                result[current] += "\n"
            result[current] += line

    return result
