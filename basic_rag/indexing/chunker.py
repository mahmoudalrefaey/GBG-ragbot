
from basic_rag.ingestion.data_loader import load_pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=64,
    length_function=len,
    is_separator_regex=False,
)


documents, file_count = load_pdfs("data")
chunks = text_splitter.split_documents(documents)

print(f"Loaded {file_count} PDF files.")
print(f"Total documents loaded: {len(documents)}")
print(f"Total chunks created: {len(chunks)}")

with open("chunks_preview.txt", "w", encoding="utf-8") as f:
    f.write(f"Total chunks: {len(chunks)}\n")
    f.write("="*50 + "\n\n")
    
    for i, chunk in enumerate(chunks):
        f.write(f"--- Chunk {i+1} ---\n")
        f.write(f"Length: {len(chunk.page_content)} characters\n")
        f.write(f"Metadata: {chunk.metadata}\n")
        f.write(f"Content:\n{chunk.page_content}\n")
        f.write("\n" + "="*50 + "\n\n")
        
print("Chunks saved to 'chunks_preview.txt'")