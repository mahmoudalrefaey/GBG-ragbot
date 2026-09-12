import os

from dotenv import load_dotenv
from advanced_rag.retrieval.base import retrieve
from advanced_rag.retrieval.reranking import rerank_documents


CRAG_THRESHOLD = float(
    os.environ.get(
        "CRAG_THRESHOLD",
        "0.0",
    )
)
load_dotenv()
RAG_N_RESULTS = int(os.getenv("RAG_N_RESULTS", "5"))


def crag_retrieve(
    query: str,
    candidate_k: int = 10,
    n_results: int = RAG_N_RESULTS,
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

    best_score = float(ranked[0][1])

    if best_score >= CRAG_THRESHOLD:
        return [
            document
            for document, _ in ranked
        ]

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
        return_scores=True,
    )

    if not corrected_ranked:
        return [
            document
            for document, _ in ranked
        ]

    corrected_best_score = float(
        corrected_ranked[0][1]
    )

    if corrected_best_score > best_score:
        return [
            document
            for document, _ in corrected_ranked
        ]

    return [
        document
        for document, _ in ranked
    ]