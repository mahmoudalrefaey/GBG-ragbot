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


def decompose_query(query: str):
    prompt = f"""
Break the following question into smaller independent
questions.

Only decompose when the question contains multiple
information needs.

If the question is already simple, return the original
question as one line.

Return one question per line.
Do not number them.
Do not answer them.

Question:
{query}
"""

    response = llm.invoke(prompt)

    questions = [
        line.strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    return questions[:4] or [query]


def decomposition_retrieve(
    query: str,
    n_results: int = RAG_N_RESULTS,
):
    sub_questions = decompose_query(query)

    result_lists = []

    for sub_question in sub_questions:
        documents = retrieve(
            sub_question,
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