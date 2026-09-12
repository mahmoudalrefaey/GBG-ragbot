import os

from dotenv import load_dotenv
from basic_rag.retrieval.retriever import retrieve_similar

load_dotenv()
RAG_N_RESULTS = int(os.getenv("RAG_N_RESULTS", "5"))


def retrieve(
    query: str,
    n_results: int = RAG_N_RESULTS,
    where: dict | None = None,
):
    return retrieve_similar(
        query=query,
        n_results=n_results,
        where=where,
    )


def documents_to_text(documents):
    return "\n\n---\n\n".join(
        item["document"]
        for item in documents
    )