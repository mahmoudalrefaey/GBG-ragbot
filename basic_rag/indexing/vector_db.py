import chromadb
import numpy as np
from typing import List

from langchain_core.documents import Document


chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_or_create_collection(
    name="gbg_collection",
    metadata={"hnsw:space": "cosine"},
)


def add_embeddings_to_chromadb(
    embeddings: np.ndarray,
    chunks: List[Document],
):
    ids = []
    documents = []
    metadatas = []
    embeddings_list = []

    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        chunk_id = f"doc_{i}"

        ids.append(chunk_id)
        documents.append(chunk.page_content)

        metadatas.append({
            "chunk_id": chunk_id,
            "source": (
            str(chunk.metadata.get("source", "unknown"))
            .replace("\\", "/")
            .split("/")[-1]
        ),
            "page": chunk.metadata.get("page", "unknown"),
            "chunk_index": i,
        })

        embeddings_list.append(embedding.tolist())

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings_list,
    )

    print(f"Added {len(ids)} documents to ChromaDB")
    print(f"Total documents in collection: {collection.count()}")


def clear_collection():
    if collection.count() > 0:
        ids = collection.get(include=[])["ids"]
        collection.delete(ids=ids)
        print("Collection cleared")