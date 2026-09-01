
from basic_rag.ingestion.data_loader import load_pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_pdfs(directory: str):
    documents, file_count = load_pdfs(directory)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=150,
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Loaded {file_count} PDF files.")
    print(f"Total documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")
    
    return chunks