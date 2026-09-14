"""
Legacy extractor functions.

The main pipeline now uses analyze_transcript() from summarize.py,
which generates title, summary, action items, decisions, and questions
in ONE Mistral request.

These functions are retained for compatibility with older code.
"""

from core.summarize import analyze_transcript


def extract_action_items(transcript: str) -> str:
    return analyze_transcript(transcript)["action_items"]


def extract_key_decisions(transcript: str) -> str:
    return analyze_transcript(transcript)["key_decisions"]


def extract_questions(transcript: str) -> str:
    return analyze_transcript(transcript)["open_questions"]
