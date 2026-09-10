from basic_rag.retrieval.retriever import retrieve_similar


def retrieve(query: str, n_results: int = 10):
    return retrieve_similar(
        query=query,
        n_results=n_results,
    )


def documents_to_text(documents):
    return "\n\n---\n\n".join(
        item["document"]
        for item in documents
    )