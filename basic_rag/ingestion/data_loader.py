from pathlib import Path

from langchain_core.documents import Document

from basic_rag.ingestion.pdf_processor import process_pdf


def load_pdfs(directory: str):
    documents = []
    file_count = 0

    for pdf_path in Path(directory).glob("*.pdf"):

        print(f"\nProcessing: {pdf_path.name}")

        pages = process_pdf(pdf_path)

        for page in pages:

            documents.append(
                Document(
                    page_content=page["text"],
                    metadata={
                        "source": str(pdf_path),
                        "page": page["page"],
                    },
                )
            )

        file_count += 1

        print(
            f"Extracted {len(pages)} pages"
        )

    return documents, file_count