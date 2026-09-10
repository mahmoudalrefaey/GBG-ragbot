from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=256,
    api_key=os.environ.get("GROQ_API_KEY"),
)


def rewrite_query(query: str) -> str:
    prompt = f"""
Rewrite the following user query into a clear, standalone
search query for retrieving information from Arabic bank documents.

Keep the original meaning.
Do not answer the question.
Return only the rewritten query.

Query:
{query}
"""

    response = llm.invoke(prompt)

    return response.content.strip()