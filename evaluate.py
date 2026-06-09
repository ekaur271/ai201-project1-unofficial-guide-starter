"""
Evaluation harness for The Unofficial Guide.

Runs the 5 test questions from planning.md through the full pipeline and prints,
for each: the question, the expected answer, the retrieved chunks (with distance
scores), and — if a Groq API key is configured — the generated answer.

Retrieval is always evaluated (no API key needed). Generation is evaluated only
when GROQ_API_KEY is set, so you can verify retrieval quality before wiring the
LLM. Fill in the accuracy judgments in README.md after reviewing the output.

Run:  python evaluate.py
"""

from config import GROQ_API_KEY
from retriever import retrieve, get_collection

# 5 test questions with ground-truth answers grounded in the collected reviews.
TEST_QUESTIONS = [
    {
        "question": "Which professor should I take for Software 1 (CSE 2221)?",
        "expected": "Naomi Zweben — reviewers say she cares about student success, "
                    "makes an intimidating weed-out class feel manageable, posts lecture "
                    "recordings, and writes fair exams aligned with class material.",
    },
    {
        "question": "What do students say about KT Vandergriff's exams?",
        "expected": "Very tough with little/no exam practice provided; students must "
                    "self-study from slides. Opinions split: kind and engaging in person, "
                    "but rough tests even with a curve.",
    },
    {
        "question": "Is Luan Duong a good choice for Operating Systems (CSE 2431)?",
        "expected": "Yes — called a 'gem'/'GOAT' of the CSE dept, patient explanations, "
                    "lots of extra credit (~13%), fair-but-hard exams. Lectures can be dry "
                    "and he has an accent students get used to.",
    },
    {
        "question": "Which CSE professor gives the most useful feedback?",
        "expected": "KT Vandergriff (insightful lab feedback on fixing mistakes) and Joe "
                    "Barker's TAs (good HW feedback) are mentioned. This is a harder, "
                    "comparative question — see failure-case analysis in the README.",
    },
    {
        "question": "What is the parking situation near the CSE building?",
        "expected": "OUT OF SCOPE — no document covers parking. The system should refuse: "
                    "'I don't have enough information on that...'",
    },
]


def main():
    if get_collection().count() == 0:
        print("Vector store is empty. Run `python app.py` once to ingest, then retry.")
        return

    use_llm = bool(GROQ_API_KEY) and GROQ_API_KEY != "your_key_here"
    if use_llm:
        from generator import generate_response
    else:
        print("⚠️  No GROQ_API_KEY set — evaluating RETRIEVAL ONLY (generation skipped).\n")

    for i, item in enumerate(TEST_QUESTIONS, start=1):
        print("=" * 78)
        print(f"Q{i}: {item['question']}")
        print(f"Expected: {item['expected']}")
        print("-" * 78)

        retrieved = retrieve(item["question"])
        print("Retrieved chunks:")
        for r in retrieved:
            snippet = r["text"].replace("\n", " ")
            print(f"  [{r['distance']:.3f}] {r['source']}: {snippet[:110]}...")

        if use_llm:
            result = generate_response(item["question"], retrieved)
            print("\nSystem answer:")
            print(f"  {result['answer']}")
            print(f"  Sources: {result['sources']}")
        print()


if __name__ == "__main__":
    main()
