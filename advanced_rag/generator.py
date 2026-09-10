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
    route = route_query(question)

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