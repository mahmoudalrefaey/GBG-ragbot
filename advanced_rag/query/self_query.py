import json
import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from advanced_rag.retrieval.base import retrieve


load_dotenv()


llm = ChatGroq(
    model=os.environ.get("LIGHT_MODEL"),
    temperature=0,
    max_tokens=256,
    api_key=os.environ.get("GROQ_API_KEY"),
)


def parse_self_query(query: str):
    prompt = f"""
Convert the following user question into JSON.

Use this exact structure:

{{
    "search_query": "...",
    "source": null,
    "page": null
}}

Rules:

- search_query must contain the actual information need.
- source should only be used when the user explicitly specifies a document/source.
- page should only be used when the user explicitly specifies a page number.
- page must be an integer.
- source should contain only the filename if a filename is specified.
- Do not invent metadata.

Return ONLY valid JSON.

Question:
{query}
"""

    response = llm.invoke(prompt)

    content = response.content.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return {
            "search_query": query,
            "source": None,
            "page": None,
        }

    return {
        "search_query": result.get("search_query") or query,
        "source": result.get("source"),
        "page": result.get("page"),
    }


def _normalize_source(source: str):
    if not source:
        return None

    source = source.strip()

    source = source.replace("\\", "/")

    return source.split("/")[-1].lower()


def _build_where(
    source=None,
    page=None,
):
    conditions = []

    if source:
        source_name = _normalize_source(source)

        data = retrieve(
            query="",
            n_results=1,
        )

        if data:
            conditions.append(
                {
                    "source": {
                        "$eq": source_name
                    }
                }
            )

    if page is not None:
        try:
            page_number = int(page)

            conditions.append(
                {
                    "page": {
                        "$eq": page_number
                    }
                }
            )
        except (TypeError, ValueError):
            pass

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {
        "$and": conditions
    }


def self_query_retrieve(
    query: str,
    n_results: int = 10,
):
    parsed = parse_self_query(query)

    search_query = parsed["search_query"]
    source = parsed["source"]
    page = parsed["page"]

    where = None

    if page is not None:
        try:
            page_number = int(page)
            where = {
                "page": {
                    "$eq": page_number
                }
            }
        except (TypeError, ValueError):
            where = None

    if source:
        source_name = _normalize_source(source)

        if where:
            where = {
                "$and": [
                    where,
                    {
                        "source": {
                            "$eq": source_name
                        }
                    },
                ]
            }
        else:
            where = {
                "source": {
                    "$eq": source_name
                }
            }

    documents = retrieve(
        search_query,
        n_results=n_results,
        where=where,
    )

    if documents:
        return documents

    return retrieve(
        search_query,
        n_results=n_results,
    )