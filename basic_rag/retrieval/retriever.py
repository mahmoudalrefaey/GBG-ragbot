from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever

from basic_rag.indexing.vector_db import collection
from basic_rag.indexing.embedder import get_query_embedding


class ChromaDenseRetriever(BaseRetriever):
    k: int = 10

    def _get_relevant_documents(self, query: str, **_) -> List[Document]:
        embedding = get_query_embedding(query)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=self.k,
        )

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
    weights: List[float] = [0.6, 0.4],
    k: int = 60,
) -> List[Document]:
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

    return [documents[doc_id] for doc_id in ranked_ids]


def _build_bm25_retriever(k: int = 10) -> BM25Retriever:
    data = collection.get()

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


_dense_retriever = ChromaDenseRetriever(k=10)
_bm25_retriever = _build_bm25_retriever(k=10)


def retrieve_similar(
    query: str,
    n_results: int = 5,
) -> List[Dict]:

    dense_docs = _dense_retriever.invoke(query)
    bm25_docs = _bm25_retriever.invoke(query)

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


def get_collection_info() -> Dict:
    return {
        "name": collection.name,
        "count": collection.count(),
        "metadata": collection.metadata,
    }