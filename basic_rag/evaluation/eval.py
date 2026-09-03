import os
import json
from pathlib import Path

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    GEval,
)
from deepeval.models import GeminiModel

from basic_rag.generation.generator import generate_answer

judge = GeminiModel(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

DATASET_PATH = Path("basic_rag/evaluation/dataset.json")

# ============================================================
# DeepEval Metrics
# ============================================================

context_relevancy = ContextualRelevancyMetric(
    threshold=0.7,
    include_reason=True,
    model=judge,
)


# 2. Faithfulness
#
# Evaluates whether the generated answer is supported
# by the retrieved context.
faithfulness = FaithfulnessMetric(
    threshold=0.7,
    include_reason=True,
    model=judge,
)


# 3. Answer Relevance
#
# Evaluates whether the generated answer actually
# addresses the user's question.
answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    include_reason=True,
    model=judge,
)


# 4. Correctness
#
# Evaluates the generated answer against the
# trusted expected answer from the dataset.
correctness = GEval(
    name="Correctness",

    evaluation_steps=[
        "Compare the actual output against the expected output.",
        "Determine whether the factual information in the actual output "
        "matches the expected output.",
        "Penalize factual contradictions.",
        "Penalize important missing information.",
        "Do not penalize differences in wording when the meaning is equivalent.",
        "Treat paraphrases with the same factual meaning as correct.",
    ],

    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],

    threshold=0.7,
    model=judge,
)


# ============================================================
# Load Evaluation Dataset
# ============================================================

def load_dataset():
    """
    Load the evaluation dataset.

    Expected JSON structure:

    {
        "dataset": [
            {
                "id": "Q001",
                "input": "...",
                "expected_output": "...",
                "category": "...",
                "difficulty": "...",
                "source": "...",
                "page": 5,
                "evidence": "...",
                "requires_human_review": false
            }
        ],
        "metadata": {...}
    }
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "dataset" not in data:
        raise ValueError(
            "Invalid dataset format. Expected a top-level 'dataset' key."
        )

    return data["dataset"]


# ============================================================
# Build DeepEval Test Cases
# ============================================================

def build_test_cases():
    """
    Run the real RAG pipeline for every question and convert
    the result into a DeepEval LLMTestCase.
    """

    dataset = load_dataset()

    test_cases = []

    print("=" * 70)
    print(f"Evaluation dataset: {len(dataset)} questions")
    print("=" * 70)

    for index, item in enumerate(dataset, start=1):

        question_id = item.get("id", f"Q{index:03d}")
        question = item["input"]
        expected_output = item.get("expected_output", "")

        print()
        print("-" * 70)
        print(f"[{index}/{len(dataset)}] {question_id}")
        print(f"Question: {question}")

        # ----------------------------------------------------
        # Run your actual RAG application
        # ----------------------------------------------------

        answer, retrieved_context = generate_answer(
            question,
            return_context=True,
        )

        print(f"Answer: {answer}")
        print(f"Retrieved chunks: {len(retrieved_context)}")

        # ----------------------------------------------------
        # Create DeepEval test case
        # ----------------------------------------------------

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_output,
            retrieval_context=retrieved_context,
        )

        test_cases.append(test_case)

    return test_cases


def run_evaluation():

    test_cases = build_test_cases()

    print()
    print("=" * 70)
    print("Running DeepEval metrics")
    print("=" * 70)
    print()

    results = evaluate(
        test_cases=test_cases,
        metrics=[
            context_relevancy,
            faithfulness,
            answer_relevancy,
            correctness,
        ],
    )

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("GBG RAG Evaluation")
    print("=" * 70)
    print()
    print(f"Judge model: gemini-2.0-flash")
    print(f"Dataset: {DATASET_PATH}")
    print()

    run_evaluation()
