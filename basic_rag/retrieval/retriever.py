from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever

from basic_rag.indexing.vector_db import collection
from basic_rag.indexing.embedder import get_query_embedding


class ChromaDenseRetriever(BaseRetriever):
    k: int = 10
    where: Optional[dict] = None

    def _get_relevant_documents(
        self,
        query: str,
        **_,
    ) -> List[Document]:
        embedding = get_query_embedding(query)

        kwargs = {
            "query_embeddings": [embedding],
            "n_results": self.k,
        }

        if self.where:
            kwargs["where"] = self.where

        results = collection.query(**kwargs)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        return [
            Document(
                page_content=doc,
                metadata=metadatas[i] or {},
            )
            for i, doc in enumerate(documents)
        ]


def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    weights: Optional[List[float]] = None,
    k: int = 60,
) -> List[Document]:
    if weights is None:
        weights = [0.6, 0.4]

    scores = {}
    documents = {}

    for weight, results in zip(weights, result_lists):
        for rank, doc in enumerate(results):
            doc_id = (
                doc.metadata.get("chunk_id")
                or doc.metadata.get("id")
                or doc.page_content
            )

            documents[doc_id] = doc

            scores[doc_id] = scores.get(doc_id, 0.0) + (
                weight / (k + rank + 1)
            )

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        documents[doc_id]
        for doc_id in ranked_ids
    ]


_bm25_retriever = None
_bm25_count = -1


def _build_bm25_retriever(k: int = 10) -> BM25Retriever:
    data = collection.get(
        include=["documents", "metadatas"],
    )

    documents = [
        Document(
            page_content=doc,
            metadata=meta or {},
        )
        for doc, meta in zip(
            data["documents"],
            data["metadatas"],
        )
    ]

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k

    return retriever


def _get_bm25_retriever(k: int = 10) -> BM25Retriever:
    global _bm25_retriever
    global _bm25_count

    current_count = collection.count()

    if (
        _bm25_retriever is None
        or _bm25_count != current_count
    ):
        _bm25_retriever = _build_bm25_retriever(k=k)
        _bm25_count = current_count

    _bm25_retriever.k = k

    return _bm25_retriever


def _filter_documents(
    documents: List[Document],
    where: Optional[dict] = None,
) -> List[Document]:
    if not where:
        return documents

    filtered = []

    for document in documents:
        metadata = document.metadata

        matches = True

        for key, value in where.items():
            if isinstance(value, dict):
                if "$eq" in value:
                    if metadata.get(key) != value["$eq"]:
                        matches = False
                        break

            elif metadata.get(key) != value:
                matches = False
                break

        if matches:
            filtered.append(document)

    return filtered


def retrieve_similar(
    query: str,
    n_results: int = 5,
    where: Optional[dict] = None,
) -> List[Dict]:
    dense_retriever = ChromaDenseRetriever(
        k=max(n_results * 3, 10),
        where=where,
    )

    dense_docs = dense_retriever.invoke(query)

    bm25_retriever = _get_bm25_retriever(
        k=max(n_results * 3, 10),
    )

    bm25_docs = bm25_retriever.invoke(query)

    bm25_docs = _filter_documents(
        bm25_docs,
        where,
    )

    hybrid_docs = reciprocal_rank_fusion(
        result_lists=[
            dense_docs,
            bm25_docs,
        ],
        weights=[0.6, 0.4],
    )

    return [
        {
            "document": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in hybrid_docs[:n_results]
    ]


def retrieve_by_embedding(
    embedding,
    n_results: int = 10,
    where: Optional[dict] = None,
):
    kwargs = {
        "query_embeddings": [embedding],
        "n_results": n_results,
    }

    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return [
        {
            "document": doc,
            "metadata": metadatas[i] or {},
        }
        for i, doc in enumerate(documents)
    ]


def get_collection_info() -> Dict:
    return {
        "name": collection.name,
        "count": collection.count(),
        "metadata": collection.metadata,
    }