import os
import re
import unicodedata
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever

from basic_rag.indexing.vector_db import collection
from basic_rag.indexing.embedder import get_query_embedding

load_dotenv()
RAG_N_RESULTS = int(os.getenv("RAG_N_RESULTS", "5"))


# ── Arabic-aware tokenizer for BM25 ────────────────────────

_DIACRITICS = re.compile(
    "["
    "\u0610-\u061A"  # Small signs
    "\u064B-\u065F"  # Harakat (fatha, damma, kasra …)
    "\u0670"         # Superscript alef
    "\u06D6-\u06DC"  # Small high signs
    "\u06DF-\u06E4"  # More small signs
    "\u06E7\u06E8"   # Small high yeh/noon
    "\u06EA-\u06ED"  # More marks
    "]+"
)

_ALEF_VARIANTS = re.compile("[إأآٱ]")

_TATWEEL = "\u0640"

_TOKEN_SPLIT = re.compile(r"[^\u0621-\u064A\u0660-\u06690-9a-zA-Z]+")


def arabic_preprocess(text: str) -> List[str]:
    """
    Lightweight Arabic normalization + tokenization for BM25.

    Steps:
      1. Unicode NFC normalization
      2. Remove diacritics (tashkeel)
      3. Normalize alef variants  →  ا
      4. Normalize taa marbuta  ة  →  ه
      5. Remove tatweel  ـ
      6. Strip definite article  ال  from token start
      7. Split on non-Arabic/non-alphanumeric characters
    """
    text = unicodedata.normalize("NFC", text)

    text = _DIACRITICS.sub("", text)

    text = _ALEF_VARIANTS.sub("ا", text)

    text = text.replace("ة", "ه")

    text = text.replace(_TATWEEL, "")

    tokens = _TOKEN_SPLIT.split(text)

    result = []
    for token in tokens:
        if not token:
            continue
        # Strip definite article ال
        if token.startswith("ال") and len(token) > 2:
            token = token[2:]
        if token:
            result.append(token)

    return result


# ── Dense retriever ────────────────────────────────────────

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


# ── Reciprocal Rank Fusion ──────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    weights: Optional[List[float]] = None,
    k: int = 60,
) -> List[Document]:
    if weights is None:
        weights = [0.5, 0.5]

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


# ── BM25 retriever (Arabic-aware, cached) ──────────────────

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

    retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=arabic_preprocess,
    )
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


# ── Document filtering ──────────────────────────────────────

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


# ── Main retrieval functions ────────────────────────────────

def retrieve_similar(
    query: str,
    n_results: int = RAG_N_RESULTS,
    where: Optional[dict] = None,
) -> List[Dict]:
    candidate_k = max(n_results * 5, 20)

    dense_retriever = ChromaDenseRetriever(
        k=candidate_k,
        where=where,
    )

    dense_docs = dense_retriever.invoke(query)

    bm25_retriever = _get_bm25_retriever(
        k=candidate_k,
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
        weights=[0.5, 0.5],
    )

    return [
        {
            "document": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in hybrid_docs[:n_results]
    ]


def retrieve_with_details(
    query: str,
    n_results: int = RAG_N_RESULTS,
    where: Optional[dict] = None,
) -> dict:
    """
    Like retrieve_similar but returns detailed info
    for the Streamlit UI (dense ranks, BM25 ranks,
    final RRF scores).
    """
    candidate_k = max(n_results * 5, 20)

    dense_retriever = ChromaDenseRetriever(
        k=candidate_k,
        where=where,
    )
    dense_docs = dense_retriever.invoke(query)

    bm25_retriever = _get_bm25_retriever(
        k=candidate_k,
    )
    bm25_docs = bm25_retriever.invoke(query)
    bm25_docs = _filter_documents(bm25_docs, where)

    # Build rank maps for logging
    dense_ranks = {}
    for rank, doc in enumerate(dense_docs):
        doc_id = (
            doc.metadata.get("chunk_id")
            or doc.page_content[:60]
        )
        dense_ranks[doc_id] = rank + 1

    bm25_ranks = {}
    for rank, doc in enumerate(bm25_docs):
        doc_id = (
            doc.metadata.get("chunk_id")
            or doc.page_content[:60]
        )
        bm25_ranks[doc_id] = rank + 1

    # RRF with score tracking
    weights = [0.5, 0.5]
    k_rrf = 60
    scores = {}
    documents = {}

    for weight, results in zip(
        weights, [dense_docs, bm25_docs]
    ):
        for rank, doc in enumerate(results):
            doc_id = (
                doc.metadata.get("chunk_id")
                or doc.metadata.get("id")
                or doc.page_content
            )
            documents[doc_id] = doc
            scores[doc_id] = scores.get(
                doc_id, 0.0
            ) + (weight / (k_rrf + rank + 1))

    ranked_ids = sorted(
        scores, key=scores.get, reverse=True,
    )

    result_docs = []
    details = []

    for doc_id in ranked_ids[:n_results]:
        doc = documents[doc_id]
        result_docs.append({
            "document": doc.page_content,
            "metadata": doc.metadata,
        })
        details.append({
            "chunk_id": doc_id,
            "dense_rank": dense_ranks.get(doc_id),
            "bm25_rank": bm25_ranks.get(doc_id),
            "rrf_score": round(scores[doc_id], 6),
            "source": doc.metadata.get("source", "?"),
            "page": doc.metadata.get("page", "?"),
        })

    return {
        "documents": result_docs,
        "details": details,
        "dense_candidate_count": len(dense_docs),
        "bm25_candidate_count": len(bm25_docs),
    }


def retrieve_by_embedding(
    embedding,
    n_results: int = RAG_N_RESULTS,
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
