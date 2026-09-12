# Evaluation

Run the evaluation with:

```powershell
.\.venv\Scripts\python.exe -m basic_rag.evaluation.evaluator --limit 1
```

The judge model is loaded from `JUDGE_MODEL_NAME` in the repository `.env`.
The evaluator passes DeepEval's Pydantic schemas to Groq using strict JSON
Schema structured outputs. `openai/gpt-oss-120b` and
`openai/gpt-oss-20b` are the Groq models documented as supporting strict
structured outputs, so `openai/gpt-oss-120b` is an appropriate judge model for
this evaluator.
The evaluator runs requests serially and does not assume undocumented Groq
quotas. It starts with conservative `EVAL_RPM`, `EVAL_RPD`, `EVAL_TPM`, and
`EVAL_TPD` budgets, learns remaining values and reset windows from Groq
`x-ratelimit-*` response headers, honors `retry-after`, and retries HTTP 429
responses with exponential backoff and jitter. Configure lower budgets if
another process shares the same Groq organization. Set `EVAL_*` values only
when they are confirmed on the account Limits page.
The judge completion default is 2048 tokens because DeepEval's contextual
relevance metric can return a list of verdicts; lower values can cause Groq
JSON mode to stop before producing valid JSON. Override it with
`EVAL_JUDGE_MAX_TOKENS` if the account's token quota requires a smaller value.
If the remaining quota would require waiting more than one hour, the run
pauses cleanly and saves the completed samples as a `partial` report instead
of raising a traceback. Run the evaluator again after the Groq reset window.

Each result includes the question, generated answer, ground truth, references,
all four DeepEval metrics with scores and reasons, retrieved chunks, source
pages, selected advanced strategy, latency, and cost metadata. The current
production pipeline exposes estimated token counts rather than provider usage
for its final response, so cost is calculated from those estimates. Override
`EVAL_INPUT_PRICE_PER_1K` and `EVAL_OUTPUT_PRICE_PER_1K` when the applicable
model pricing differs.
