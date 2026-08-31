import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

data_dir = Path("data")

def load_pdfs(directory: str):
    """
    Load a PDF file and return its content as a list of documents.

    Args:
        directory (str): The path to the PDF file.
    """
    
    documents = []
    file_count = 0
    
    for pdf_path in Path(directory).glob("*.pdf"):
        loader = PyMuPDFLoader(str(pdf_path))
        documents.extend(loader.load())
        file_count += 1
    
    return documents, file_count