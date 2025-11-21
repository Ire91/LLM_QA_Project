import json
from typing import Tuple

from llm_client import build_prompt, preprocess_question, query_llm


def display_banner() -> None:
    print("=" * 72)
    print("NLP Question-and-Answering CLI powered by an LLM API")
    print("Type 'quit' or 'exit' to close the application.")
    print("=" * 72)


def run_cli() -> None:
    display_banner()

    while True:
        user_question = input("\nEnter your question: ").strip()
        if user_question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if not user_question:
            print("⚠️  Please enter a valid question.")
            continue

        try:
            preprocessing = preprocess_question(user_question)
            prompt = build_prompt(preprocessing["processed"])
            answer, raw_response = query_llm(prompt)
        except Exception as exc:
            print(f"An error occurred: {exc}")
            continue

        print("\n--- Processed Question ---")
        print(preprocessing["processed"])

        print("\n--- Answer ---")
        print(answer)

        debug_choice = input(
            "\nWould you like to view the raw LLM response JSON? (y/N): "
        ).strip()
        if debug_choice.lower() == "y":
            print(json.dumps(raw_response, indent=2))


if __name__ == "__main__":
    run_cli()


