from sentence_transformers import SentenceTransformer
from basic_rag.indexing.chunker import chunk_pdfs
from basic_rag.indexing.vector_db import add_embeddings_to_chromadb
import numpy as np

def generate_and_index_embeddings(
    data_path: str = "data",
    model_name: str = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
    batch_size: int = 32,
    should_index: bool = True
):
    print("Loading and chunking documents...")
    model = SentenceTransformer(model_name)
    chunks = chunk_pdfs(data_path)
    chunks_text = [doc.page_content for doc in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(
        chunks_text,
        convert_to_tensor=False,
        batch_size=batch_size,
        show_progress_bar=True
    )

    print(f"Embedding Shape: {embeddings.shape}")
    
    if should_index:
        print("Indexing to ChromaDB...")
        add_embeddings_to_chromadb(embeddings, chunks, batch_size=1000)
    
    return embeddings, chunks

def get_query_embedding(
    query: str,
    model_name: str = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
):
    model = SentenceTransformer(model_name)
    return model.encode(query, convert_to_tensor=False)