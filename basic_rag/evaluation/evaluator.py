"""
Simple RAG evaluation using DeepEval with OpenRouter as the judge LLM.

Reuses the existing RAG pipeline (generator + retriever) and the existing
evaluation dataset. Default run evaluates only 2 questions for a fast test.

Run with limit=2 (default) for a fast check, or limit=None for the full set.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

from basic_rag.generation.generator import generate_answer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "basic_rag" / "evaluation"
DEFAULT_DATASET_PATH = EVAL_DIR / "full_eval_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JUDGE_MODEL_NAME = "minimax/minimax-m3:free"


CORRECTNESS_RUBRIC = """
You are evaluating the correctness of a RAG system's answer.

Compare the actual_output to the expected_output and judge whether
the actual answer conveys the same meaning, with attention to:

- factual correctness
- missing important information
- contradictions
- numbers, dates, frequencies, quantities
- proper names and entities
- policy or procedural requirements

Arabic paraphrases that preserve the same meaning MUST be considered correct.
Only mark the answer incorrect if important information is wrong, missing,
or contradicted. Stylistic or wording differences are acceptable.
"""


def load_judge():
    """Build the OpenRouter judge model used by all metrics."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment or .env")

    return OpenRouterModel(
        model=JUDGE_MODEL_NAME,
        api_key=api_key,
    )


def load_questions(limit, dataset_path=None):
    """Load the evaluation questions from the dataset JSON."""
    path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("dataset", [])
    if limit is not None:
        questions = questions[:limit]
    return questions


def build_test_case(question_obj, answer, context_chunks):
    """Build a DeepEval LLMTestCase for a single question."""
    return LLMTestCase(
        input=question_obj["input"],
        actual_output=answer,
        expected_output=question_obj["expected_output"],
        retrieval_context=context_chunks,
    )


def build_metrics(judge):
    """Build the four evaluation metrics, all sharing the same judge."""
    correctness = GEval(
        name="Correctness",
        criteria=CORRECTNESS_RUBRIC,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        threshold=0.5,
    )

    return [
        ("Context Relevance", ContextualRelevancyMetric(model=judge, threshold=0.5)),
        ("Faithfulness", FaithfulnessMetric(model=judge, threshold=0.5)),
        ("Answer Relevance", AnswerRelevancyMetric(model=judge, threshold=0.5)),
        ("Correctness", correctness),
    ]


def run_evaluation(limit=2, dataset_path=None):
    """Run the full evaluation loop. limit=None means evaluate everything."""
    questions = load_questions(limit, dataset_path)

    print("RAG EVALUATION")
    print(f"Dataset: {dataset_path or DEFAULT_DATASET_PATH.name}")
    print(f"Dataset size: {len(questions)}")
    print(f"Judge model: {JUDGE_MODEL_NAME}\n")

    judge = load_judge()
    metrics = build_metrics(judge)

    per_question_results = []
    metric_sums = {name: 0.0 for name, _ in metrics}
    metric_counts = {name: 0 for name, _ in metrics}

    for idx, q in enumerate(questions, start=1):
        qid = q.get("id", f"Q{idx:03d}")
        question = q["input"]
        expected = q["expected_output"]

        print(f"[{idx}/{len(questions)}] {qid}")
        print(f"Question: {question}")

        answer, context_chunks = generate_answer(
            question=question,
            n_results=4,
            return_context=True,
        )

        print(f"Answer: {answer}")

        test_case = build_test_case(q, answer, context_chunks)

        scores = {}
        for name, metric in metrics:
            try:
                metric.measure(test_case, _show_indicator=False)
                score = metric.score if metric.score is not None else 0.0
            except Exception as e:
                print(f"  ! {name} failed: {e}")
                score = 0.0
            scores[name] = score
            metric_sums[name] += score
            metric_counts[name] += 1

            print(f"{name}: {score:.3f}")

        per_question_results.append(
            {
                "id": qid,
                "input": question,
                "expected_output": expected,
                "actual_output": answer,
                "retrieval_context": context_chunks,
                "category": q.get("category"),
                "difficulty": q.get("difficulty"),
                "source": q.get("source"),
                "page": q.get("page"),
                "scores": scores,
            }
        )

        print()

    averages = {
        name: (metric_sums[name] / metric_counts[name])
        if metric_counts[name] > 0
        else 0.0
        for name in metric_sums
    }

    print("AVERAGE SCORES")
    for name, avg in averages.items():
        print(f"{name}: {avg:.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"results_{timestamp}.json"

    payload = {
        "judge_model": JUDGE_MODEL_NAME,
        "dataset_size": len(questions),
        "averages": averages,
        "questions": per_question_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    run_evaluation(limit=None)
