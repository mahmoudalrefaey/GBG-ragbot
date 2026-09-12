
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from basic_rag.ingestion.data_loader import load_pdfs


def chunk_pdfs(directory: str):
    documents, file_count = load_pdfs(directory)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Loaded {file_count} PDF files.")
    print(f"Total documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")
    
    return chunks


# def save_chunks_to_json(chunks, output_path: str | Path):
#     """Save chunk text and metadata for inspecting generated chunks."""
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     payload = [
#         {
#             "chunk_index": index,
#             "text": chunk.page_content,
#             "metadata": chunk.metadata,
#         }
#         for index, chunk in enumerate(chunks)
#     ]

#     output_path.write_text(
#         json.dumps(payload, ensure_ascii=False, indent=2),
#         encoding="utf-8",
#     )

#     print(f"Saved {len(payload)} chunks to {output_path}")


# if __name__ == "__main__":
#     generated_chunks = chunk_pdfs("data")
#     save_chunks_to_json(generated_chunks, "chunks.json")