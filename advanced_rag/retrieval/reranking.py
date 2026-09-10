from sentence_transformers import CrossEncoder

from advanced_rag.retrieval.base import retrieve


RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


reranker = CrossEncoder(
    RERANKER_MODEL,
    device="cpu",
)


def rerank_documents(
    query: str,
    documents,
    n_results: int = 4,
    return_scores: bool = False,
):
    if not documents:
        return []

    pairs = [
        [query, item["document"]]
        for item in documents
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    ranked = ranked[:n_results]

    if return_scores:
        return ranked

    return [
        document
        for document, _ in ranked
    ]


def reranked_retrieve(
    query: str,
    candidate_k: int = 15,
    n_results: int = 4,
):
    documents = retrieve(
        query,
        n_results=candidate_k,
    )

    return rerank_documents(
        query,
        documents,
        n_results=n_results,
    )