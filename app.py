import json
from flask import Flask, render_template, request

from llm_client import LLMClientError, build_prompt, preprocess_question, query_llm


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "question": "",
        "processed": "",
        "answer": "",
        "raw_response": "",
        "error": "",
    }

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        context["question"] = question

        if not question:
            context["error"] = "Please enter a question."
            return render_template("index.html", **context)

        try:
            preprocessing = preprocess_question(question)
            prompt = build_prompt(preprocessing["processed"])
            answer, raw = query_llm(prompt)
            context["processed"] = preprocessing["processed"]
            context["answer"] = answer
            context["raw_response"] = json.dumps(raw, indent=2)
        except (ValueError, LLMClientError) as exc:
            context["error"] = str(exc)

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)


