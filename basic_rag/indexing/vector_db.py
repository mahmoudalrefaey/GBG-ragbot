import chromadb
from typing import List, Dict
import numpy as np
from langchain_core.documents import Document

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(
    name="gbg_collection",
    metadata={"hnsw:space": "cosine"}
)

def add_embeddings_to_chromadb(
    embeddings: np.ndarray,
    chunks: List[Document],
    batch_size: int = 1000
):
    ids = []
    documents = []
    metadatas = []
    embeddings_list = []
    
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        ids.append(f"doc_{i}")
        documents.append(chunk.page_content)
        metadatas.append({
            "source": chunk.metadata.get("source", "unknown"),
            "chunk_index": i
        })
        embeddings_list.append(embedding.tolist())
    
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_emb = embeddings_list[i:i+batch_size]
        
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_emb
        )
        
        print(f"Added {len(batch_ids)} documents to ChromaDB")
    
    print(f"Total documents in collection: {collection.count()}")

def retrieve_similar(
    query_embedding: List[float],
    n_results: int = 5,
    where_filter: Dict = None
):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter
    )
    
    retrieved = []
    for i, doc in enumerate(results['documents'][0]):
        retrieved.append({
            'document': doc,
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]  # Lower = more similar
        })
    
    return retrieved

def get_collection_info():
    return {
        'name': collection.name,
        'count': collection.count(),
        'metadata': collection.metadata
    }

def clear_collection():
    if collection.count() > 0:
        ids = collection.get(include=[])['ids']
        collection.delete(ids=ids)
        print("Collection cleared")