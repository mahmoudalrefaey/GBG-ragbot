import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


llm = ChatGroq(
    model=os.environ.get("LIGHT_MODEL"),
    temperature=0,
    max_tokens=256,
    api_key=os.environ.get("GROQ_API_KEY"),
)


def rewrite_query(query: str) -> str:
    prompt = f"""
Rewrite the following user query into a clear, standalone
search query for retrieving information from Arabic bank documents.

Keep the original meaning.
Preserve names, numbers, dates, frequencies, and conditions.

Do not answer the question.
Return only the rewritten query.

Query:
{query}
"""

    response = llm.invoke(prompt)

    rewritten = response.content.strip()

    return rewritten or query