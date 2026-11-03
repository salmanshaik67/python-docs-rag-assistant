"""Measure retrieval quality on a small hand-written set of questions.

For each question we know which docs page should answer it, and report
hit@1 (the best chunk comes from that page) and hit@3 (any of the top 3 does).
"""

from rag import Retriever

EVAL_SET = [
    ("How do I read a JSON file?", "tutorial/inputoutput"),
    ("How do I serialize a dict to JSON?", "library/json"),
    ("What is a virtual environment?", "tutorial/venv"),
    ("How do I handle exceptions?", "tutorial/errors"),
    ("How do list comprehensions work?", "tutorial/datastructures"),
    ("How do I format dates?", "library/datetime"),
    ("How do I define a class?", "tutorial/classes"),
    ("How do I import a module?", "tutorial/modules"),
    ("How can I join file paths?", "library/os.path"),
    ("How do I match a regular expression?", "library/re"),
    ("How do I count items with a Counter?", "library/collections"),
    ("How do for loops and range work?", "tutorial/controlflow"),
    ("How do I list files in a directory with Path?", "library/pathlib"),
    ("How do I chain iterators together?", "library/itertools"),
]


def main() -> None:
    retriever = Retriever()
    hit1 = hit3 = 0
    for question, expected in EVAL_SET:
        pages = [r["source"] for r in retriever.search(question, k=3)]
        top1 = expected in pages[0] if pages else False
        top3 = any(expected in p for p in pages)
        hit1 += top1
        hit3 += top3
        mark = "ok " if top1 else ("~  " if top3 else "MISS")
        print(f"{mark} {question}")
    n = len(EVAL_SET)
    print(f"\nhit@1: {hit1}/{n} ({hit1 / n:.0%})   hit@3: {hit3}/{n} ({hit3 / n:.0%})")


if __name__ == "__main__":
    main()
