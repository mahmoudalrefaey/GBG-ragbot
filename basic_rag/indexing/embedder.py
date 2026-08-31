from sentence_transformers import SentenceTransformer
from basic_rag.indexing.chunker import chunk_pdfs
import numpy as np

def generate_embeddings(
    data_path: str = "data",
    model_name: str = "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
    batch_size: int = 32
):
    model = SentenceTransformer(model_name)
    chunks = chunk_pdfs(data_path)
    chunks_text = [doc.page_content for doc in chunks]

    embeddings = model.encode(
        chunks_text,
        convert_to_tensor=False,
        batch_size=batch_size,
        show_progress_bar=True
    )

    print(f"Embedding Shape: {embeddings.shape}")
    return embeddings, chunks