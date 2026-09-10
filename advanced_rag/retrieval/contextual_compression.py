from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from advanced_rag.retrieval.base import retrieve

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=512,
    api_key=os.environ.get("GROQ_API_KEY"),
)


def compress_document(query: str, document: str):
    prompt = f"""
Extract only the parts of the document that are relevant
to answering the question.

Do not add information.
Do not answer the question.
If nothing is relevant, return an empty string.

Question:
{query}

Document:
{document}
"""

    response = llm.invoke(prompt)

    return response.content.strip()


def compressed_retrieve(
    query: str,
    candidate_k: int = 8,
    n_results: int = 4,
):
    documents = retrieve(
        query,
        n_results=candidate_k,
    )

    compressed = []

    for item in documents:
        text = compress_document(
            query,
            item["document"],
        )

        if text:
            compressed.append({
                "document": text,
                "metadata": item["metadata"],
            })

    return compressed[:n_results]