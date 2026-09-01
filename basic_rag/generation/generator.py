from langchain_ollama import ChatOllama

from basic_rag.generation.prompts import RAG_PROMPT
from basic_rag.retrieval.retriever import retrieve_similar


llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    num_ctx=4096
)


def generate_answer(
    question: str,
    n_results: int = 4,
) -> str:

    retrieved = retrieve_similar(
        query=question,
        n_results=n_results,
    )

    if not retrieved:
        return "There is no documentation available to answer your question. Please provide more context or check the documentation."

    # for item in retrieved:
    #     print("\n--- RETRIEVED DOCUMENT ---")
    #     print("Source:", item["metadata"].get("source"))
    #     print("Chunk:", item["metadata"].get("chunk_index"))
    #     print("Distance:", item["distance"])
    #     print(item["document"])
    
    context_parts = []

    for item in retrieved:
        document = item["document"]
        metadata = item.get("metadata", {})
        distance = item.get("distance")

        source = metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index", "unknown")

        context_parts.append(
            f"""Source: {source}
Chunk: {chunk_index}
Distance: {distance}

{document}"""
        )

    context = "\n\n---\n\n".join(context_parts)

    messages = RAG_PROMPT.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    return response.content