from sentence_transformers import SentenceTransformer
from basic_rag.indexing.chunker import chunk_pdfs
import numpy as np

model = SentenceTransformer('Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2')

chunks = chunk_pdfs("data")
chunks_text = [doc.page_content for doc in chunks]

# Generate embeddings (default of Matryoshka is 768 dims)
embeddings = model.encode(
    chunks_text,
    convert_to_tensor=False,
    batch_size=32,
    show_progress_bar=True
)

print(f"Embedding Shape: {embeddings.shape}")