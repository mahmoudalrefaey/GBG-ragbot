import os
from pathlib import Path
import pymupdf
from PIL import Image
import unicodedata
import pytesseract
from langchain_core.documents import Document

data_dir = Path("data")

def load_pdfs(directory: str):
    documents = []
    file_count = 0
    
    for pdf_path in Path(directory).glob("*.pdf"):
        print(f"Processing: {pdf_path.name}")
        
        doc = pymupdf.open(str(pdf_path))
        full_text = []
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Render page to image
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            
            # Extract text with OCR (Arabic)
            text = pytesseract.image_to_string(image, lang='ara')
            
            # Basic cleanup
            text = unicodedata.normalize('NFC', text)
            text = text.replace('\u200E', '').replace('\u200F', '').replace('\u061C', '')
            
            if text.strip():
                full_text.append(text)
        
        doc.close()
        
        documents.append(Document(
            page_content="\n\n".join(full_text),
            metadata={"source": str(pdf_path)}
        ))
        
        file_count += 1
        print(f"Extracted {len(full_text)} pages\n")
    
    return documents, file_count