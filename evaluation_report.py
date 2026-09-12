"""Simple Streamlit page for the latest DeepEval report."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import streamlit as st


RESULTS_DIR = Path(__file__).resolve().parent / "basic_rag" / "evaluation" / "results"

st.set_page_config(
    page_title="Evaluation Report",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    :root {
        --bg-primary: #0f0f13;
        --bg-card: #1e1e2e;
        --accent-secondary: #a78bfa;
        --text-primary: #e4e4ef;
        --text-secondary: #9090a8;
        --text-muted: #5e5e76;
        --border: rgba(124, 106, 239, 0.15);
    }
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }
    .report-header {
        text-align: center;
        padding: 1rem 0 1.25rem;
    }
    .report-header h1 {
        margin-bottom: 0.25rem;
        background: linear-gradient(135deg, var(--accent-secondary), #c084fc, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .report-header p {
        color: var(--text-muted);
        margin: 0;
    }
    .card {
        background: linear-gradient(135deg, var(--bg-card), rgba(30, 30, 46, 0.55));
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin: 0.5rem 0;
    }
    .card-label {
        color: var(--text-muted);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .score {
        color: var(--accent-secondary);
        font-size: 1.35rem;
        font-weight: 700;
    }
    .arabic {
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', sans-serif;
        line-height: 1.8;
    }
    .chunk {
        background: #0f0f13;
        border: 1px solid var(--border);
        border-radius: 9px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    .chunk-meta {
        color: var(--text-muted);
        font-size: 0.72rem;
        margin-bottom: 0.35rem;
    }
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def latest_report() -> tuple[dict[str, Any] | None, str | None]:
    try:
        files = sorted(
            RESULTS_DIR.glob("results*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError as error:
        return None, f"Could not inspect evaluation results: {error}"

    if not files:
        return None, None

    try:
        with files[0].open(encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f"Could not read the evaluation report: {error}"

    if not isinstance(report, dict):
        return None, "The evaluation report has an unexpected format."
    return report, None


def text(value: Any, fallback: str = "Not available") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def card(label: str, value: Any, *, arabic: bool = False) -> None:
    class_name = "arabic" if arabic else ""
    safe_value = html.escape(text(value))
    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">{html.escape(label)}</div>
            <div class="{class_name}">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(metrics: Any) -> None:
    if not isinstance(metrics, dict) or not metrics:
        st.info("No metric results are available for this sample.")
        return

    columns = st.columns(min(4, len(metrics)))
    for column, (name, detail) in zip(columns, metrics.items()):
        with column:
            score = detail.get("score") if isinstance(detail, dict) else detail
            score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else text(score)
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">{html.escape(str(name))}</div>
                    <div class="score">{html.escape(score_text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    for name, detail in metrics.items():
        reason = detail.get("reason") if isinstance(detail, dict) else None
        if reason:
            with st.expander(f"{name} reason"):
                st.write(reason)


def render_sources(sources: Any) -> None:
    if not isinstance(sources, list) or not sources:
        st.info("No sources are available.")
        return
    for source in sources:
        if not isinstance(source, dict):
            continue
        document = text(source.get("document", source.get("source")))
        page = text(source.get("page"))
        card("Source", f"{document} · Page {page}")


def render_metadata_cards(value: Any, empty_message: str) -> None:
    if not isinstance(value, dict) or not value:
        card("Details", value if value else empty_message)
        return

    items = list(value.items())
    for start in range(0, len(items), 3):
        columns = st.columns(min(3, len(items) - start))
        for column, (key, item) in zip(columns, items[start:start + 3]):
            with column:
                if isinstance(item, bool):
                    display = "Yes" if item else "No"
                elif isinstance(item, float):
                    display = f"{item:.4f}"
                else:
                    display = item
                card(str(key).replace("_", " "), display)


def render_chunks(chunks: Any) -> None:
    if not isinstance(chunks, list) or not chunks:
        st.info("No retrieved chunks are available.")
        return
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            continue
        source = text(chunk.get("source"))
        page = text(chunk.get("page"))
        document = html.escape(text(chunk.get("document"), ""))
        st.markdown(
            f"""
            <div class="chunk">
                <div class="chunk-meta">Chunk {index} · {html.escape(source)} · Page {html.escape(page)}</div>
                <div class="arabic">{document}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown(
    """
    <div class="report-header">
        <h1>📊 Evaluation Report</h1>
        <p>DeepEval results for the GBG RAG system</p>
    </div>
    """,
    unsafe_allow_html=True,
)

report, error = latest_report()
if error:
    st.error(error)
    st.stop()
if report is None:
    st.info("📋 Evaluation report not yet available. Run the evaluator to generate it.")
    st.stop()

averages = report.get("averages")
if isinstance(averages, dict) and averages:
    st.subheader("Average scores")
    render_metrics(averages)

results = report.get("results")
if not isinstance(results, list) or not results:
    st.info("The report exists, but it contains no evaluation samples yet.")
    st.stop()

st.subheader("Evaluation samples")
for index, sample in enumerate(results, start=1):
    if not isinstance(sample, dict):
        st.warning(f"Sample {index} could not be displayed.")
        continue

    with st.expander(f"Sample {index}: {text(sample.get('question'))}", expanded=index == 1):
        question, category = st.columns(2)
        with question:
            card("Question", sample.get("question"), arabic=True)
        with category:
            card("Category", sample.get("category"))

        answer, truth = st.columns(2)
        with answer:
            card("Generated answer", sample.get("generated_answer", sample.get("answer")), arabic=True)
        with truth:
            card("Ground truth", sample.get("ground_truth"), arabic=True)

        card("References", sample.get("references"), arabic=True)

        with st.expander("Metrics", expanded=True):
            render_metrics(sample.get("metrics"))

        sources = sample.get("sources")
        source_count = len(sources) if isinstance(sources, list) else 0
        with st.expander(f"Sources ({source_count})"):
            render_sources(sources)

        strategy = sample.get("advanced_rag_strategies_used")
        with st.expander("RAG strategies used"):
            if strategy:
                render_metadata_cards(strategy, "No strategy details are available.")
            else:
                st.info("No advanced RAG strategies were recorded.")

        chunks = sample.get("retrieved_chunks")
        chunk_count = len(chunks) if isinstance(chunks, list) else 0
        with st.expander(f"Retrieved chunks ({chunk_count})"):
            render_chunks(chunks)

        cost, timing = st.columns(2)
        with cost:
            with st.expander("Cost"):
                render_metadata_cards(
                    sample.get("cost"),
                    "Cost information is not available.",
                )
        with timing:
            with st.expander("Time"):
                render_metadata_cards(
                    sample.get("time"),
                    "Timing information is not available.",
                )
