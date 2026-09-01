import chromadb
from typing import List, Dict
from basic_rag.indexing.embedder import get_query_embedding

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(
    name="gbg_collection",
    metadata={"hnsw:space": "cosine"}
)

def retrieve_similar(
    query: str,
    n_results: int = 5,
    where_filter: Dict = None,
    model_name: str = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
):
    query_embedding = get_query_embedding(query)
    
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
            'distance': results['distances'][0][i] 
        })
    
    return retrieved

def get_collection_info() -> Dict:
    """Get collection stats."""
    return {
        'name': collection.name,
        'count': collection.count(),
        'metadata': collection.metadata
    }