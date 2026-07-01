"""
Unit evaluation of the Critic Agent (Phase 3) in isolation.

Feeds hand-labelled (context, answer) pairs directly into critic_node,
bypassing retrieval/generation, so the critic's hallucination-detection
accuracy can be measured independently of retrieval quality. Half the
cases are faithful paraphrases of the context (label: grounded); half
contain an injected fabricated or contradictory claim (label: hallucinated).

Reports precision/recall/F1 for hallucination detection plus a confusion matrix.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Agent.nodes.critic import critic_node

DATASET = Path(__file__).parent / "critic_cases.jsonl"
OUT = Path(__file__).parent / "critic_eval_results.json"


def load_cases():
    with open(DATASET) as f:
        return [json.loads(line) for line in f if line.strip()]


def run():
    cases = load_cases()
    results = []

    tp = fp = tn = fn = 0

    for case in cases:
        state = {
            "question": case["question"],
            "context": case["context"],
            "answer": case["answer"],
            "plan": {"mode": "explain"},
            "reflection_count": 0,
        }
        out = critic_node(state)

        predicted_grounded = out.get("critic_passed", True) and out.get("answer") == case["answer"]
        # critic_node doesn't expose "grounded" directly; critic_passed=True with an
        # unchanged answer means it judged the answer grounded on the first pass.
        actual_hallucinated = not case["grounded"]
        predicted_hallucinated = not predicted_grounded

        if predicted_hallucinated and actual_hallucinated:
            tp += 1
        elif predicted_hallucinated and not actual_hallucinated:
            fp += 1
        elif not predicted_hallucinated and not actual_hallucinated:
            tn += 1
        else:
            fn += 1

        row = {
            "id": case["id"],
            "gold_grounded": case["grounded"],
            "predicted_grounded": predicted_grounded,
            "correct": predicted_hallucinated == actual_hallucinated,
        }
        results.append(row)
        print(f"[{case['id']}] gold_grounded={case['grounded']} "
              f"predicted_grounded={predicted_grounded} "
              f"{'OK' if row['correct'] else 'WRONG'}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    n = len(cases)
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    print("\n=== Critic hallucination-detection metrics ===")
    print(f"Cases: {n}")
    print(f"Confusion matrix: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Accuracy:  {accuracy:.0%}")
    print(f"Precision: {precision:.0%}")
    print(f"Recall:    {recall:.0%}")
    print(f"F1:        {f1:.0%}")
    print(f"\nFull results written to {OUT}")


if __name__ == "__main__":
    run()
