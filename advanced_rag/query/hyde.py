import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from basic_rag.indexing.embedder import get_query_embedding
from basic_rag.retrieval.retriever import retrieve_by_embedding


load_dotenv()
RAG_N_RESULTS = int(os.getenv("RAG_N_RESULTS", "5"))


llm = ChatGroq(
    model=os.environ.get("LIGHT_MODEL"),
    temperature=0,
    max_tokens=384,
    api_key=os.environ.get("GROQ_API_KEY"),
)


def generate_hypothetical_document(query: str):
    prompt = f"""
Write a short hypothetical document passage that could
contain the answer to the following question.

Use terminology likely to appear in Arabic bank documents.
Focus on factual content and relevant concepts.

Do not mention that the passage is hypothetical.
Do not explain your reasoning.

Question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content.strip()


def hyde_retrieve(
    query: str,
    n_results: int = RAG_N_RESULTS,
):
    hypothetical_document = generate_hypothetical_document(
        query
    )

    if not hypothetical_document:
        return []

    embedding = get_query_embedding(
        hypothetical_document
    )

    return retrieve_by_embedding(
        embedding,
        n_results=n_results,
    )