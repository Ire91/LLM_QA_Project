import os
import re
from typing import Any, Dict, Tuple

import requests
from dotenv import load_dotenv


load_dotenv()


class LLMClientError(RuntimeError):
    """Raised when querying the LLM API fails."""


def preprocess_question(question: str) -> Dict[str, Any]:
    """Apply simple NLP preprocessing to a natural-language question."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    lowered = question.strip().lower()
    tokens = re.findall(r"\b\w+\b", lowered)
    processed_text = " ".join(tokens)

    return {
        "original": question.strip(),
        "lowercased": lowered,
        "tokens": tokens,
        "processed": processed_text,
    }


def build_prompt(processed_question: str) -> str:
    """Construct the final prompt forwarded to the LLM API."""
    instructions = (
        "You are a helpful AI tutor. "
        "Answer the student's question clearly and concisely. "
        "If the question is ambiguous, note the missing details."
    )
    return (
        f"{instructions}\n\n"
        f"Question: {processed_question}\n"
        "Answer:"
    )


def query_llm(prompt: str) -> Tuple[str, Dict[str, Any]]:
    """
    Send the prompt to the configured LLM provider and return the answer
    along with the raw JSON payload.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        base_url = os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions"
        )
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-specdec")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"
        )
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        provider = "openai"

    if not api_key:
        raise LLMClientError(
            f"Missing API key for provider '{provider}'. "
            "Set OPENAI_API_KEY or GROQ_API_KEY."
        )

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise and student-friendly AI tutor. "
                    "Explain concepts with short paragraphs and, when helpful, bullet lists."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(base_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise LLMClientError(f"HTTP error while contacting {provider}: {exc}") from exc

    try:
        answer = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError(
            f"Unexpected response structure from {provider}: {data}"
        ) from exc

    return answer, data


