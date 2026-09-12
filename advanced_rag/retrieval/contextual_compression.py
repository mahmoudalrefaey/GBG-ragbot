import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from advanced_rag.retrieval.base import retrieve
from advanced_rag.retrieval.reranking import rerank_documents


load_dotenv()
RAG_N_RESULTS = int(os.getenv("RAG_N_RESULTS", "5"))


llm = ChatGroq(
    model=os.environ.get("LIGHT_MODEL"),
    temperature=0,
    max_tokens=1024,
    api_key=os.environ.get("GROQ_API_KEY"),
)


def compress_documents(
    query: str,
    documents,
):
    if not documents:
        return []

    sections = []

    for i, item in enumerate(documents):
        sections.append(
            f"""
DOCUMENT {i}

Source: {item["metadata"].get("source", "unknown")}
Page: {item["metadata"].get("page", "unknown")}

{item["document"]}
"""
        )

    prompt = f"""
You are compressing retrieved documents for an Arabic
bank-document RAG system.

For each document:

- Keep only information relevant to the question.
- Preserve exact numbers, dates, names, conditions,
  frequencies, and procedural requirements.
- Do not add information.
- Do not answer the question.
- If a document contains nothing relevant, return EMPTY.

Return the result using exactly this format:

DOCUMENT 0
<relevant text or EMPTY>

DOCUMENT 1
<relevant text or EMPTY>

Do not change the document numbers.

Question:
{query}

Retrieved documents:
{"".join(sections)}
"""

    response = llm.invoke(prompt)

    lines = response.content.splitlines()

    compressed = {}
    current_document = None
    current_text = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("DOCUMENT "):
            if current_document is not None:
                compressed[current_document] = "\n".join(
                    current_text
                ).strip()

            try:
                current_document = int(
                    stripped.replace("DOCUMENT ", "")
                )
            except ValueError:
                current_document = None

            current_text = []
            continue

        if current_document is not None:
            current_text.append(line)

    if current_document is not None:
        compressed[current_document] = "\n".join(
            current_text
        ).strip()

    result = []

    for index, item in enumerate(documents):
        text = compressed.get(index, "").strip()

        if not text or text.upper() == "EMPTY":
            continue

        result.append({
            "document": text,
            "metadata": item["metadata"],
        })

    return result


def compressed_retrieve(
    query: str,
    candidate_k: int = 10,
    n_results: int = RAG_N_RESULTS,
):
    documents = retrieve(
        query,
        n_results=candidate_k,
    )

    ranked = rerank_documents(
        query,
        documents,
        n_results=n_results,
    )

    return compress_documents(
        query,
        ranked,
    )