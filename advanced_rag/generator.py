from basic_rag.generation.generator import llm, RAG_PROMPT
from basic_rag.generation.generator import generate_answer

from advanced_rag.router import route_query

from advanced_rag.query.query_rewriting import rewrite_query
from advanced_rag.query.multi_query import multi_query_retrieve
from advanced_rag.query.decomposition import decomposition_retrieve
from advanced_rag.query.hyde import hyde_retrieve
from advanced_rag.query.self_query import self_query_retrieve

from advanced_rag.retrieval.reranking import reranked_retrieve
from advanced_rag.retrieval.contextual_compression import compressed_retrieve
from advanced_rag.retrieval.crag import crag_retrieve


def format_context(documents):
    parts = []

    for item in documents:
        metadata = item.get("metadata", {})

        parts.append(
            f"""Source: {metadata.get("source", "unknown")}
Page: {metadata.get("page", "unknown")}

{item["document"]}"""
        )

    return "\n\n---\n\n".join(parts)


def generate_advanced_answer(
    question: str,
    return_context: bool = False,
):
    route = route_query(question)

    if route == "basic":
        return generate_answer(
            question,
            return_context=return_context,
        )

    if route == "rewrite":
        rewritten = rewrite_query(question)

        from advanced_rag.retrieval.base import retrieve

        documents = retrieve(rewritten, n_results=10)

    elif route == "multi_query":
        documents = multi_query_retrieve(question)

    elif route == "decomposition":
        documents = decomposition_retrieve(question)

    elif route == "hyde":
        documents = hyde_retrieve(question)

    elif route == "self_query":
        documents = self_query_retrieve(question)

    elif route == "reranking":
        documents = reranked_retrieve(question)

    elif route == "compression":
        documents = compressed_retrieve(question)

    elif route == "crag":
        documents = crag_retrieve(question)

    else:
        return generate_answer(
            question,
            return_context=return_context,
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