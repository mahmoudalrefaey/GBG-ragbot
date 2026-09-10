import time

from basic_rag.generation.generator import (
    llm,
    RAG_PROMPT,
    generate_answer,
)

from advanced_rag.router import route_query

from advanced_rag.query.query_rewriting import (
    rewrite_query,
)
from advanced_rag.query.multi_query import (
    multi_query_retrieve,
)
from advanced_rag.query.decomposition import (
    decomposition_retrieve,
)
from advanced_rag.query.hyde import (
    hyde_retrieve,
)
from advanced_rag.query.self_query import (
    self_query_retrieve,
)

from advanced_rag.retrieval.base import retrieve
from advanced_rag.retrieval.reranking import (
    reranked_retrieve,
)
from advanced_rag.retrieval.contextual_compression import (
    compressed_retrieve,
)
from advanced_rag.retrieval.crag import (
    crag_retrieve,
)


# Maps route names to human-readable technique descriptions
ROUTE_DESCRIPTIONS = {
    "basic": "Basic Hybrid Retrieval (Dense + BM25 + RRF)",
    "rewrite": "Query Rewriting → Hybrid Retrieval",
    "multi_query": "Multi-Query Expansion → Hybrid Retrieval → RRF Merge",
    "decomposition": "Query Decomposition → Hybrid Retrieval → RRF Merge",
    "hyde": "HyDE (Hypothetical Document Embedding) → Dense Retrieval",
    "self_query": "Self-Query (Metadata Filtering) → Hybrid Retrieval",
    "reranking": "Hybrid Retrieval → Cross-Encoder Reranking",
    "compression": "Hybrid Retrieval → Reranking → Contextual Compression",
    "crag": "Corrective RAG (Retrieval → Reranking → Corrective Loop)",
}


def format_context(documents):
    parts = []

    for item in documents:
        metadata = item.get(
            "metadata",
            {},
        )

        parts.append(
            f"""Source: {metadata.get("source", "unknown")}
Page: {metadata.get("page", "unknown")}

{item["document"]}"""
        )

    return "\n\n---\n\n".join(parts)


def _retrieve_by_route(
    question: str,
    route: str,
):
    if route == "basic":
        return retrieve(
            question,
            n_results=4,
        )

    if route == "rewrite":
        rewritten = rewrite_query(question)

        return retrieve(
            rewritten,
            n_results=4,
        )

    if route == "multi_query":
        return multi_query_retrieve(
            question,
            n_results=4,
        )

    if route == "decomposition":
        return decomposition_retrieve(
            question,
            n_results=4,
        )

    if route == "hyde":
        return hyde_retrieve(
            question,
            n_results=4,
        )

    if route == "self_query":
        return self_query_retrieve(
            question,
            n_results=4,
        )

    if route == "reranking":
        return reranked_retrieve(
            question,
            candidate_k=15,
            n_results=4,
        )

    if route == "compression":
        return compressed_retrieve(
            question,
            candidate_k=10,
            n_results=4,
        )

    if route == "crag":
        return crag_retrieve(
            question,
            candidate_k=10,
            n_results=4,
        )

    return []


def generate_advanced_answer(
    question: str,
    return_context: bool = False,
):
    route_result = route_query(question)
    route = route_result["route"]

    documents = _retrieve_by_route(
        question,
        route,
    )

    if not documents:
        return generate_answer(
            question,
            return_context=return_context,
        )

    context = format_context(documents)

    messages = RAG_PROMPT.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    answer = response.content

    if return_context:
        return answer, documents

    return answer


def generate_advanced_answer_with_metadata(
    question: str,
) -> dict:
    """
    Full pipeline execution returning everything the UI needs:
      - answer: str
      - route: str
      - route_reason: str
      - technique: str (human-readable)
      - documents: list of retrieved docs
      - execution_time_s: float
      - estimated_input_tokens: int
      - estimated_output_tokens: int
      - fallback_used: bool
    """
    start = time.time()

    # Step 1: Route
    route_start = time.time()
    route_result = route_query(question)
    route_time = time.time() - route_start

    route = route_result["route"]
    reason = route_result["reason"]

    # Step 2: Retrieve
    retrieval_start = time.time()
    documents = _retrieve_by_route(question, route)
    retrieval_time = time.time() - retrieval_start

    fallback_used = False

    if not documents:
        fallback_used = True
        retrieval_start2 = time.time()
        documents = retrieve(question, n_results=4)
        retrieval_time += time.time() - retrieval_start2

    # Step 3: Generate
    generation_start = time.time()

    context = format_context(documents)

    messages = RAG_PROMPT.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)
    generation_time = time.time() - generation_start

    answer = response.content

    total_time = time.time() - start

    # Estimate tokens (rough: ~1 token per 3 chars for Arabic)
    input_text = context + question
    est_input_tokens = len(input_text) // 3
    est_output_tokens = len(answer) // 3

    return {
        "answer": answer,
        "route": route,
        "route_reason": reason,
        "technique": ROUTE_DESCRIPTIONS.get(
            route, route
        ),
        "documents": documents,
        "execution_time_s": round(total_time, 2),
        "route_time_s": round(route_time, 2),
        "retrieval_time_s": round(retrieval_time, 2),
        "generation_time_s": round(generation_time, 2),
        "estimated_input_tokens": est_input_tokens,
        "estimated_output_tokens": est_output_tokens,
        "fallback_used": fallback_used,
    }