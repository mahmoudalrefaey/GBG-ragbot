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


def decompose_query(query: str):
    prompt = f"""
Break the following question into smaller independent questions.

Only decompose the question when necessary.
Each question should represent one information need.

Return one question per line.
Do not number them.
Do not answer them.

Question:
{query}
"""

    response = llm.invoke(prompt)

    return [
        line.strip()
        for line in response.content.splitlines()
        if line.strip()
    ]


def decomposition_retrieve(query: str, n_results: int = 5):
    sub_questions = decompose_query(query)

    all_documents = {}

    for sub_question in sub_questions:
        documents = retrieve(
            sub_question,
            n_results=n_results,
        )

        for item in documents:
            key = item["document"]

            if key not in all_documents:
                all_documents[key] = item

    return list(all_documents.values())