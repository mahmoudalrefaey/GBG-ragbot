from sentence_transformers import SentenceTransformer

from basic_rag.indexing.chunker import chunk_pdfs
from basic_rag.indexing.vector_db import add_embeddings_to_chromadb


EMBEDDING_MODEL_NAME = (
    "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
)


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME, 
    device= "cpu"
    )


def generate_and_index_embeddings(
    data_path: str = "data",
):
    """
    Load documents, create chunks, generate embeddings,
    and store them in ChromaDB.
    """

    print("Loading and chunking documents...")

    chunks = chunk_pdfs(data_path)

    if not chunks:
        print("No documents found.")
        return

    print(f"Total chunks created: {len(chunks)}")

    chunks_text = [
        chunk.page_content
        for chunk in chunks
    ]

    print("Generating embeddings...")

    embeddings = embedding_model.encode(
        chunks_text,
        convert_to_tensor=False,
        show_progress_bar=True,
    )

    print(f"Embedding Shape: {embeddings.shape}")

    print("Indexing to ChromaDB...")

    add_embeddings_to_chromadb(
        embeddings,
        chunks,
    )


def get_query_embedding(query: str):
    """
    Generate an embedding for a user query.

    The embedding model is loaded once at application startup
    and reused for all queries.
    """

    return embedding_model.encode(
        query,
        convert_to_tensor=False,
    )