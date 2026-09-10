import json

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from basic_rag.retrieval.retriever import retrieve_similar

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=512,
    api_key=os.environ.get("GROQ_API_KEY"),
)


def parse_self_query(query: str):
    prompt = f"""
Convert the following question into JSON.

Return exactly:

{{
    "search_query": "...",
    "source": null,
    "page": null
}}

Use source or page only if the user explicitly specifies them.

Question:
{query}
"""

    response = llm.invoke(prompt)

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {
            "search_query": query,
            "source": None,
            "page": None,
        }


def self_query_retrieve(query: str, n_results: int = 10):
    parsed = parse_self_query(query)

    documents = retrieve_similar(
        query=parsed["search_query"],
        n_results=n_results,
    )

    if parsed["source"] is not None:
        documents = [
            item
            for item in documents
            if item["metadata"].get("source") == parsed["source"]
        ]

    if parsed["page"] is not None:
        documents = [
            item
            for item in documents
            if item["metadata"].get("page") == parsed["page"]
        ]

    return documents