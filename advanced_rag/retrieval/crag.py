from advanced_rag.retrieval.base import retrieve
from advanced_rag.retrieval.reranking import rerank_documents


def crag_retrieve(
    query: str,
    candidate_k: int = 10,
    n_results: int = 4,
    threshold: float = 0.0,
):
    documents = retrieve(
        query,
        n_results=candidate_k,
    )

    ranked = rerank_documents(
        query,
        documents,
        n_results=n_results,
        return_scores=True,
    )

    if not ranked:
        return []

    best_score = ranked[0][1]

    if best_score >= threshold:
        return [
            document
            for document, _ in ranked
        ]

    # Corrective fallback:
    # search again using a rewritten query.
    from advanced_rag.query.query_rewriting import rewrite_query

    rewritten_query = rewrite_query(query)

    corrected_documents = retrieve(
        rewritten_query,
        n_results=candidate_k,
    )

    corrected_ranked = rerank_documents(
        rewritten_query,
        corrected_documents,
        n_results=n_results,
    )

    return corrected_ranked