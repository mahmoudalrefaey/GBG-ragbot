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


VALID_ROUTES = {
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


ROUTER_PROMPT = """
You are a routing classifier for an Arabic bank-document RAG system.

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

basic:
Use for normal factual questions that can likely be answered
with standard hybrid retrieval.

rewrite:
Use when the query is vague, ambiguous, poorly phrased,
or missing important search wording.

multi_query:
Use when different wording or terminology could retrieve
different relevant documents for the same information need.

decomposition:
Use only when the question contains multiple independent
information needs that should be searched separately.

hyde:
Use for conceptual or semantic questions where generating
a hypothetical relevant passage could improve dense retrieval.

self_query:
Use only when the user explicitly specifies metadata such as
a document name or page number.

reranking:
Use when the query is specific but several similar candidate
documents are likely and better ranking is useful.

compression:
Use when retrieved documents are likely to contain long,
irrelevant sections.

crag:
Use when retrieval is likely to be uncertain and corrective
retrieval may be useful.

Prefer BASIC whenever an advanced technique is not clearly
necessary.

Return ONLY valid JSON:

{"route":"basic"}

User query:
"""


def route_query(query: str) -> str:
    try:
        response = router_llm.invoke(
            ROUTER_PROMPT + query
        )

        result = json.loads(
            response.content
        )

        route = result.get(
            "route",
            "basic",
        )

    except (
        json.JSONDecodeError,
        TypeError,
        AttributeError,
    ):
        return "basic"

    if route not in VALID_ROUTES:
        return "basic"

    return route