import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from advanced_rag.retrieval.base import retrieve


load_dotenv()
RAG_N_RESULTS = int(os.getenv("RAG_N_RESULTS", "5"))


llm = ChatGroq(
    model=os.environ.get("LIGHT_MODEL"),
    temperature=0,
    max_tokens=256,
    api_key=os.environ.get("GROQ_API_KEY"),
)


def generate_queries(
    query: str,
    num_queries: int = 3,
):
    prompt = f"""
Generate {num_queries} different search queries for the
following question.

The queries must preserve the same information need while
using different wording.

Return exactly one query per line.
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

    queries = queries[:num_queries]

    if query not in queries:
        queries.insert(0, query)

    return queries[:num_queries]


def multi_query_retrieve(
    query: str,
    n_results: int = RAG_N_RESULTS,
):
    queries = generate_queries(query)

    result_lists = []

    for generated_query in queries:
        documents = retrieve(
            generated_query,
            n_results=n_results,
        )

        result_lists.append(documents)

    scores = {}
    documents = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            metadata = item.get("metadata", {})

            doc_id = (
                metadata.get("chunk_id")
                or item["document"]
            )

            documents[doc_id] = item

            scores[doc_id] = scores.get(doc_id, 0.0) + (
                1.0 / (60 + rank + 1)
            )

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        documents[doc_id]
        for doc_id in ranked_ids[:n_results]
    ]