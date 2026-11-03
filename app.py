"""Flask web app: ask a question, get the best-matching passage from the Python docs."""

from flask import Flask, jsonify, render_template, request

from rag import Retriever

app = Flask(__name__)
retriever = Retriever()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/ask")
def ask():
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify(error="Please enter a question."), 400
    if len(question) > 500:
        return jsonify(error="Question is too long (500 characters max)."), 400
    return jsonify(retriever.answer(question))


@app.get("/health")
def health():
    return jsonify(status="ok", chunks=len(retriever.chunks))


if __name__ == "__main__":
    app.run(debug=False)
