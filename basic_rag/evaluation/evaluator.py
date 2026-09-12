"""Rate-limit-aware DeepEval evaluation for the GBG RAG pipeline.

The evaluator deliberately runs requests serially. Groq free-tier limits are
organization-level and model-specific, so the scheduler uses conservative
budgets, learns remaining limits from response headers, honors ``retry-after``,
and retries 429 responses with exponential backoff. Set the EVAL_* variables
in the environment when the account limits are known.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from langchain_groq import ChatGroq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = PROJECT_ROOT / "basic_rag" / "evaluation"
DEFAULT_DATASET = EVALUATION_DIR / "eval_dataset.json"
RESULTS_DIR = EVALUATION_DIR / "results"

load_dotenv(PROJECT_ROOT / ".env")

JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME")
if not JUDGE_MODEL_NAME:
    raise RuntimeError("JUDGE_MODEL_NAME is required in .env")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _duration_seconds(value: str | None, default: float = 60.0) -> float:
    if not value:
        return default
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", value.strip())
    if not parts:
        return default
    multipliers = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(float(amount) * multipliers[unit] for amount, unit in parts)


@dataclass
class RateLimitState:
    rpm: int
    rpd: int
    tpm: int
    tpd: int
    request_times: deque[float]
    token_times: deque[tuple[float, int]]
    day_started: float
    requests_today: int = 0
    tokens_today: int = 0
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    requests_reset_at: float | None = None
    tokens_reset_at: float | None = None


class EvaluationQuotaExhausted(RuntimeError):
    """Raised when continuing would require waiting beyond the safe limit."""


class GroqRateLimiter:
    """A conservative in-process queue for request and token quotas."""

    def __init__(self) -> None:
        now = time.monotonic()
        self.state = RateLimitState(
            rpm=_env_int("EVAL_RPM", 20),
            rpd=_env_int("EVAL_RPD", 500),
            tpm=_env_int("EVAL_TPM", 10_000),
            tpd=_env_int("EVAL_TPD", 100_000),
            request_times=deque(),
            token_times=deque(),
            day_started=now,
        )

    def _roll_windows(self, now: float) -> None:
        while self.state.request_times and now - self.state.request_times[0] >= 60:
            self.state.request_times.popleft()
        while self.state.token_times and now - self.state.token_times[0][0] >= 60:
            self.state.token_times.popleft()
        if now - self.state.day_started >= 86400:
            self.state.day_started = now
            self.state.requests_today = 0
            self.state.tokens_today = 0

    def acquire(self, estimated_tokens: int) -> None:
        while True:
            now = time.monotonic()
            self._roll_windows(now)
            request_wait = 0.0
            token_wait = 0.0

            if len(self.state.request_times) >= self.state.rpm:
                request_wait = 60 - (now - self.state.request_times[0])
            if sum(t for _, t in self.state.token_times) + estimated_tokens > self.state.tpm:
                token_wait = 60 - (now - self.state.token_times[0][0])
            if self.state.requests_today >= self.state.rpd:
                request_wait = max(request_wait, 86400 - (now - self.state.day_started))
            if self.state.tokens_today + estimated_tokens > self.state.tpd:
                token_wait = max(token_wait, 86400 - (now - self.state.day_started))

            if self.state.remaining_requests is not None and self.state.remaining_requests <= 0:
                request_wait = max(request_wait, self.state.requests_reset_at - now
                                   if self.state.requests_reset_at else 60)
            if self.state.remaining_tokens is not None and estimated_tokens > self.state.remaining_tokens:
                token_wait = max(token_wait, self.state.tokens_reset_at - now
                                 if self.state.tokens_reset_at else 60)

            wait = max(request_wait, token_wait, 0.0)
            if wait <= 0:
                self.state.request_times.append(now)
                self.state.token_times.append((now, estimated_tokens))
                self.state.requests_today += 1
                self.state.tokens_today += estimated_tokens
                return
            if wait > 3600:
                raise EvaluationQuotaExhausted(
                    f"Groq evaluation quota is exhausted; retry after "
                    f"{round(wait / 3600, 1)} hours."
                )
            time.sleep(wait + 0.05)

    def update_from_response(self, response: Any) -> None:
        metadata = getattr(response, "response_metadata", {}) or {}
        headers = metadata.get("headers", metadata.get("http_headers", {})) or {}
        headers = {str(k).lower(): v for k, v in headers.items()}
        request_limit = _optional_int(headers.get("x-ratelimit-limit-requests"))
        token_limit = _optional_int(headers.get("x-ratelimit-limit-tokens"))
        if request_limit and request_limit > 0:
            self.state.rpd = min(self.state.rpd, request_limit)
        if token_limit and token_limit > 0:
            self.state.tpm = min(self.state.tpm, token_limit)
        self.state.remaining_requests = _optional_int(headers.get("x-ratelimit-remaining-requests"))
        self.state.remaining_tokens = _optional_int(headers.get("x-ratelimit-remaining-tokens"))
        now = time.monotonic()
        request_reset = _duration_seconds(headers.get("x-ratelimit-reset-requests"), 0)
        token_reset = _duration_seconds(headers.get("x-ratelimit-reset-tokens"), 0)
        self.state.requests_reset_at = now + request_reset if request_reset else None
        self.state.tokens_reset_at = now + token_reset if token_reset else None

    def retry_delay(self, error: Exception, attempt: int) -> float:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), 3600)
            except ValueError:
                pass
        return min(2 ** attempt, 60) + random.uniform(0, 0.5)


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval adapter that routes every judge call through the limiter."""

    def __init__(self, model_name: str, limiter: GroqRateLimiter) -> None:
        self.model_name = model_name
        self.limiter = limiter
        self.llm = ChatGroq(
            model=model_name,
            temperature=0,
            max_tokens=_env_int("EVAL_JUDGE_MAX_TOKENS", 2048),
            api_key=os.getenv("GROQ_API_KEY"),
            max_retries=0,
        )

    def load_model(self) -> ChatGroq:
        return self.llm

    def generate(self, prompt: str, schema: Any | None = None) -> str:
        estimated_tokens = max(1, len(prompt) // 3) + _env_int(
            "EVAL_JUDGE_MAX_TOKENS", 2048
        )
        for attempt in range(_env_int("EVAL_MAX_RETRIES", 5) + 1):
            self.limiter.acquire(estimated_tokens)
            try:
                if schema is None:
                    response = self.llm.invoke(prompt)
                    self.limiter.update_from_response(response)
                    return str(response.content)

                # DeepEval passes a Pydantic schema. Groq strict JSON Schema
                # validation can reject some DeepEval-generated schemas before
                # generation, so use JSON mode and validate locally instead.
                response = self.llm.bind(
                    response_format={"type": "json_object"}
                ).invoke(prompt)
                self.limiter.update_from_response(response)
                parsed_json = json.loads(str(response.content))
                if hasattr(schema, "model_validate"):
                    parsed = schema.model_validate(parsed_json)
                    return parsed.model_dump_json()
                return json.dumps(parsed_json, ensure_ascii=False)
            except Exception as error:
                status_code = getattr(getattr(error, "response", None), "status_code", None)
                error_code = getattr(
                    getattr(getattr(error, "response", None), "json", lambda: {})(),
                    "get",
                    lambda _key, _default=None: None,
                )("error", {}).get("code")
                retryable = status_code == 429 or error_code == "json_validate_failed"
                if not retryable or attempt >= _env_int("EVAL_MAX_RETRIES", 5):
                    raise
                delay = self.limiter.retry_delay(error, attempt)
                if error_code == "json_validate_failed":
                    delay = min(2 ** attempt, 10) + random.uniform(0, 0.5)
                time.sleep(delay)
        raise RuntimeError("Judge request loop exited unexpectedly.")

    def supports_structured_outputs(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True

    async def a_generate(self, prompt: str, schema: Any | None = None) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name


CORRECTNESS_CRITERIA = """
Compare the actual answer with the expected answer. Judge factual correctness,
coverage of important details, numbers, dates, names, conditions, and procedural
requirements. Accept Arabic paraphrases that preserve the meaning. Penalize
missing, contradicted, or invented information, but not stylistic differences.
"""


def build_metrics(judge: GroqJudge) -> list[tuple[str, Any]]:
    return [
        ("Context Relevance", ContextualRelevancyMetric(
            model=judge, threshold=0.5, async_mode=False, include_reason=True
        )),
        ("Faithfulness", FaithfulnessMetric(
            model=judge, threshold=0.5, async_mode=False, include_reason=True
        )),
        ("Answer Relevance", AnswerRelevancyMetric(
            model=judge, threshold=0.5, async_mode=False, include_reason=True
        )),
        ("Correctness", GEval(
            name="Correctness",
            criteria=CORRECTNESS_CRITERIA,
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=judge,
            threshold=0.5,
            async_mode=False,
        )),
    ]


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Evaluation dataset must be a JSON array.")
    required = {"question", "ground_truth", "question_category", "ground_truth_reference"}
    for index, record in enumerate(records):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Dataset record {index} is missing: {sorted(missing)}")
    return records


def _chunk_source(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata", {})
    return {
        "document": item.get("document", ""),
        "source": metadata.get("source", "unknown"),
        "page": metadata.get("page", "unknown"),
    }


def _metric_result(metric: Any) -> dict[str, Any]:
    return {"score": metric.score, "reason": metric.reason}


def _estimate_cost(pipeline: dict[str, Any]) -> dict[str, Any]:
    input_tokens = int(pipeline.get("estimated_input_tokens", 0))
    output_tokens = int(pipeline.get("estimated_output_tokens", 0))
    input_rate = float(os.getenv("EVAL_INPUT_PRICE_PER_1K", "0.0005"))
    output_rate = float(os.getenv("EVAL_OUTPUT_PRICE_PER_1K", "0.0015"))
    value = (
        input_tokens * input_rate / 1000
        + output_tokens * output_rate / 1000
    )
    return {
        "value": round(value, 8),
        "currency": "USD",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_price_per_1k": input_rate,
        "output_price_per_1k": output_rate,
        "estimated": True,
    }


def _save_results(
    dataset_path: str | Path,
    results: list[dict[str, Any]],
    sums: dict[str, float],
    *,
    status: str = "complete",
    message: str | None = None,
) -> Path:
    averages = {
        name: round(total / len(results), 4) if results else 0.0
        for name, total in sums.items()
    }
    payload: dict[str, Any] = {
        "judge_model": JUDGE_MODEL_NAME,
        "dataset": str(Path(dataset_path)),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "averages": averages,
        "results": results,
    }
    if message:
        payload["message"] = message

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def evaluate(limit: int | None = None, dataset_path: str | Path = DEFAULT_DATASET) -> dict[str, Any]:
    from advanced_rag.generator import generate_advanced_answer_with_metadata

    records = load_dataset(dataset_path)
    if limit is not None:
        records = records[:limit]
    limiter = GroqRateLimiter()
    judge = GroqJudge(JUDGE_MODEL_NAME, limiter)
    metrics = build_metrics(judge)
    results: list[dict[str, Any]] = []
    sums = {name: 0.0 for name, _ in metrics}

    quota_message = None
    for index, record in enumerate(records, start=1):
        try:
            question = record["question"]
            started = time.perf_counter()
            pipeline = generate_advanced_answer_with_metadata(question)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            answer = pipeline["answer"]
            chunks = [_chunk_source(item) for item in pipeline.get("documents", [])]
            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=record["ground_truth"],
                retrieval_context=[chunk["document"] for chunk in chunks],
                completion_time=latency_ms / 1000,
            )
            metric_results = {}
            for name, metric in metrics:
                metric.measure(test_case, _show_indicator=False)
                metric_results[name] = _metric_result(metric)
                sums[name] += float(metric.score or 0)

            results.append({
                "question": question,
                "answer": answer,
                "ground_truth": record["ground_truth"],
                "category": record["question_category"],
                "references": record["ground_truth_reference"],
                "generated_answer": answer,
                "metrics": metric_results,
                "retrieved_chunks": chunks,
                "sources": [
                    {"document": chunk["source"], "page": chunk["page"]}
                    for chunk in chunks
                ],
                "advanced_rag_strategies_used": {
                    "route": pipeline.get("route"),
                    "technique": pipeline.get("technique"),
                    "fallback_used": pipeline.get("fallback_used", False),
                },
                "cost": _estimate_cost(pipeline),
                "time": {"milliseconds": latency_ms, "seconds": latency_ms / 1000},
                "question_id": record.get("question_id", f"q{index}"),
            })
            print(f"[{index}/{len(records)}] {record.get('question_id', index)}")
        except EvaluationQuotaExhausted as error:
            quota_message = str(error)
            print(f"Evaluation paused after {len(results)} samples: {quota_message}")
            break

    status = "partial" if quota_message else "complete"
    output = _save_results(
        dataset_path,
        results,
        sums,
        status=status,
        message=quota_message,
    )
    print(f"Results saved to: {output}")
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepEval over the GBG RAG dataset.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    evaluate(limit=args.limit, dataset_path=args.dataset)


if __name__ == "__main__":
    main()
