import json
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


router_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=100,
    api_key=os.environ.get("GROQ_API_KEY"),
)


ROUTER_PROMPT = """
You are a routing classifier for an Arabic document RAG system.

Choose exactly ONE route:

basic
rewrite
multi_query
decomposition
hyde
self_query
reranking
compression
crag

Rules:

- basic: simple factual question that can be answered with normal retrieval.
- rewrite: ambiguous, vague, incomplete, or poorly phrased query.
- multi_query: query may benefit from several different search formulations.
- decomposition: complex question containing multiple independent information needs.
- hyde: conceptual question where a hypothetical document would improve semantic retrieval.
- self_query: query explicitly contains metadata constraints such as source, document, or page.
- reranking: normal retrieval may return relevant candidates but ranking needs improvement.
- compression: retrieved documents are likely long or contain substantial irrelevant text.
- crag: retrieval quality may be poor or uncertain and corrective retrieval is useful.

Prefer BASIC whenever an advanced technique is not clearly necessary.

Return ONLY valid JSON:

{"route":"basic"}

User query:
"""


def route_query(query: str) -> str:
    response = router_llm.invoke(
        ROUTER_PROMPT + query
    )

    try:
        result = json.loads(response.content)
        route = result.get("route", "basic")

    except (json.JSONDecodeError, TypeError):
        return "basic"

    valid_routes = {
        "basic",
        "rewrite",
        "multi_query",
        "decomposition",
        "hyde",
        "self_query",
        "reranking",
        "compression",
        "crag",
    }

    if route not in valid_routes:
        return "basic"

    return route