"""
End-to-end evaluation of the full LangGraph pipeline.

For each labelled question in dataset.jsonl, runs the real graph (planner ->
lecture_router -> rag -> specialist agent -> synthesizer -> critic -> memory)
and checks:
  - planner mode accuracy (predicted plan['mode'] vs expected_mode)
  - lecture routing accuracy (predicted lecture_ids vs expected_lectures, in-scope only)
  - keyword grounding (whether expected keywords show up in the final answer)
  - refusal correctness on out-of-scope questions (should say it can't find the info)
  - critic_passed rate and latency

This is a lexical/behavioural proxy, not a substitute for human grading of
answer quality -- see REPORT.md Sec. 9 for that caveat.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Agent.graph import app
from rag.course_context import load_course_profile

DATASET = Path(__file__).parent / "dataset.jsonl"
OUT = Path(__file__).parent / "pipeline_eval_results.json"

REFUSAL_MARKERS = [
    "could not find",
    "not in the lecture",
    "not covered",
    "no information",
    "cannot find",
    "don't have information",
    "do not have information",
    "does not contain",
    "do not have that information",
    "not have details",
    "i do not have",
]


def load_cases():
    with open(DATASET) as f:
        return [json.loads(line) for line in f if line.strip()]


def initial_state(question):
    return {
        "question": question,
        "context": "",
        "answer": "",
        "retrieved_chunks": [],
        "plan": {},
        "lecture_route": {},
        "course_profile": load_course_profile(),
        "chat_history": [],
        "student_memory": {},
        "weak_topics": [],
        "critic_passed": True,
        "reflection_count": 0,
    }


def run():
    cases = load_cases()
    results = []

    for case in cases:
        t0 = time.time()
        result = app.invoke(initial_state(case["question"]))
        latency = time.time() - t0

        plan = result.get("plan", {})
        predicted_mode = plan.get("mode")
        predicted_lectures = sorted(result.get("lecture_route", {}).get("lecture_ids", []))
        answer = (result.get("answer") or "").lower()

        mode_correct = predicted_mode == case["expected_mode"]

        expected_lectures = sorted(case.get("expected_lectures", []))
        lecture_correct = None
        if expected_lectures:
            lecture_correct = bool(set(expected_lectures) & set(predicted_lectures))

        keyword_hit = None
        if case.get("keywords"):
            keyword_hit = all(kw.lower() in answer for kw in case["keywords"])

        refused = any(marker in answer for marker in REFUSAL_MARKERS)

        row = {
            "id": case["id"],
            "question": case["question"],
            "expected_mode": case["expected_mode"],
            "predicted_mode": predicted_mode,
            "mode_correct": mode_correct,
            "expected_lectures": expected_lectures,
            "predicted_lectures": predicted_lectures,
            "lecture_correct": lecture_correct,
            "keyword_hit": keyword_hit,
            "out_of_scope": case.get("out_of_scope", False),
            "refused": refused,
            "critic_passed": result.get("critic_passed"),
            "latency_sec": round(latency, 2),
            "answer": result.get("answer"),
        }
        results.append(row)

        status = "OK" if mode_correct else "MODE-MISMATCH"
        print(f"[{case['id']}] {status} mode={predicted_mode} lectures={predicted_lectures} "
              f"refused={refused} critic_passed={row['critic_passed']} ({latency:.1f}s)")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    summarize(results)


def summarize(results):
    n = len(results)
    mode_acc = sum(r["mode_correct"] for r in results) / n

    in_scope = [r for r in results if not r["out_of_scope"] and r["expected_lectures"]]
    lecture_acc = (
        sum(r["lecture_correct"] for r in in_scope) / len(in_scope) if in_scope else float("nan")
    )

    keyworded = [r for r in results if r["keyword_hit"] is not None]
    keyword_rate = sum(r["keyword_hit"] for r in keyworded) / len(keyworded) if keyworded else float("nan")

    oos = [r for r in results if r["out_of_scope"]]
    refusal_rate = sum(r["refused"] for r in oos) / len(oos) if oos else float("nan")

    critic_passed_rate = sum(bool(r["critic_passed"]) for r in results) / n
    avg_latency = sum(r["latency_sec"] for r in results) / n

    print("\n=== Summary ===")
    print(f"Cases: {n}")
    print(f"Planner mode accuracy: {mode_acc:.0%}")
    print(f"Lecture routing accuracy (in-scope, n={len(in_scope)}): {lecture_acc:.0%}")
    print(f"Keyword grounding rate (n={len(keyworded)}): {keyword_rate:.0%}")
    print(f"Out-of-scope refusal rate (n={len(oos)}): {refusal_rate:.0%}")
    print(f"Critic-passed rate: {critic_passed_rate:.0%}")
    print(f"Avg latency: {avg_latency:.1f}s")
    print(f"\nFull results written to {OUT}")


if __name__ == "__main__":
    run()
