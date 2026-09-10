from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from basic_rag.indexing.embedder import get_query_embedding
from basic_rag.retrieval.retriever import retrieve_by_embedding

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=512,
    api_key=os.environ.get("GROQ_API_KEY"),
)


def generate_hypothetical_document(query: str):
    prompt = f"""
Write a hypothetical answer to the following question.

The answer should contain the type of factual information
that would likely appear in the relevant bank documentation.

Do not mention that this is hypothetical.

Question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content.strip()


def hyde_retrieve(query: str, n_results: int = 10):
    hypothetical_document = generate_hypothetical_document(query)

    embedding = get_query_embedding(hypothetical_document)

    return retrieve_by_embedding(
        embedding,
        n_results=n_results,
    )