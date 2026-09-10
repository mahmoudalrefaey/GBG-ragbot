import os

from langchain_groq import ChatGroq
from basic_rag.generation.prompts import RAG_PROMPT
from basic_rag.retrieval.retriever import retrieve_similar
from dotenv import load_dotenv

load_dotenv()  

llm = ChatGroq(
    model=os.environ.get("MAIN_MODEL"),
    temperature=0,
    max_tokens=1024,
    api_key=os.environ.get("GROQ_API_KEY"),
)

def generate_answer(
    question: str,
    n_results: int = 4,
    return_context: bool = False,
):
    retrieved = retrieve_similar(
        query=question,
        n_results=n_results,
    )

    if not retrieved:
        answer = (
            "There is no documentation available to answer your question. "
            "Please provide more context or check the documentation."
        )
        if return_context:
            return answer, []
        return answer

    context_parts = []

    for item in retrieved:
        document = item["document"]
        metadata = item.get("metadata", {})

        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")

        context_parts.append(
            f"""Source: {source}
Page: {page}

{document}"""
        )

    context = "\n\n---\n\n".join(context_parts)

    messages = RAG_PROMPT.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    answer = response.content

    if return_context:
        return answer, context_parts

    return answer