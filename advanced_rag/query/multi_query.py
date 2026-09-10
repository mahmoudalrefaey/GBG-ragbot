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


def generate_queries(query: str, num_queries: int = 3):
    prompt = f"""
Generate {num_queries} different search queries for the
following question.

The queries should preserve the same information need while
using different wording.

Return one query per line.
Do not number them.
Do not answer the question.

Question:
{query}
"""

    response = llm.invoke(prompt)

    queries = [
        line.strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    return queries[:num_queries]


def multi_query_retrieve(query: str, n_results: int = 10):
    queries = generate_queries(query)

    all_documents = {}

    for generated_query in queries:
        documents = retrieve(
            generated_query,
            n_results=n_results,
        )

        for item in documents:
            key = item["document"]

            if key not in all_documents:
                all_documents[key] = item

    return list(all_documents.values())