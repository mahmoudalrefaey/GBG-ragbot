import json
import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


router_llm = ChatGroq(
    model=os.environ.get("LIGHT_MODEL"),
    temperature=0,
    max_tokens=200,
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

Choose exactly ONE route from:

basic, rewrite, multi_query, decomposition, hyde, self_query, reranking, compression, crag

Decision rules:

basic:
  Simple, clear factual question with specific keywords that
  standard hybrid retrieval can match directly.
  Example: "ما هي غرامة فقدان بطاقة الدخول؟"

rewrite:
  The query is vague, ambiguous, conversational, or uses
  informal phrasing that needs cleanup for retrieval.
  Example: "ايش الاجراء لما حد يضيع الكرت؟"

multi_query:
  The question could benefit from multiple phrasings because
  the key concept may appear under different terminology.
  Example: "ما هي إجراءات التعامل مع الأصول الراكدة؟"

decomposition:
  The question contains TWO OR MORE independent sub-questions
  that need separate retrieval.
  Example: "ما هي مدة الجرد وكم عدد أعضاء اللجنة؟"

hyde:
  Conceptual or abstract question where generating a
  hypothetical answer passage would help dense retrieval.
  Example: "كيف يتم التعامل مع المواد التالفة في المستودع؟"

self_query:
  The user explicitly mentions a document name, source,
  or page number as a filter.
  Example: "في دليل الإنذار المركزي صفحة 5، ما هي صلاحيات المدير؟"

reranking:
  A specific factual question where many similar chunks may
  be retrieved and better ranking would help find the exact one.
  Example: "من هو صاحب صلاحية الموافقة على دخول جناح الإدارة؟"

compression:
  Retrieved documents are likely long with much irrelevant
  content that should be filtered before generation.

crag:
  High uncertainty about whether retrieval will find the right
  document; corrective re-retrieval may be needed.

Return ONLY valid JSON with route and reason:

{"route":"basic","reason":"short explanation"}

User query:
"""


def _extract_json(text: str) -> dict | None:
    """Extract JSON from raw LLM output, handling
    fenced code blocks and extra text."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from fenced code block
    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.DOTALL,
    )
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } in the text
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def route_query(query: str) -> dict:
    """
    Route a query and return a dict with:
      - route: str (the selected route name)
      - reason: str (why this route was chosen)
    """
    try:
        response = router_llm.invoke(
            ROUTER_PROMPT + query
        )

        result = _extract_json(response.content)

        if result is None:
            return {
                "route": "basic",
                "reason": "Could not parse router output",
            }

        route = result.get("route", "basic")
        reason = result.get("reason", "")

    except Exception:
        return {
            "route": "basic",
            "reason": "Router error, defaulting to basic",
        }

    if route not in VALID_ROUTES:
        return {
            "route": "basic",
            "reason": f"Invalid route '{route}', defaulting to basic",
        }

    return {
        "route": route,
        "reason": reason,
    }