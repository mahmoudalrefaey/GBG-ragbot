from basic_rag.indexing.embedder import generate_and_index_embeddings
from basic_rag.retrieval.retriever import get_collection_info
from basic_rag.generation.generator import generate_answer


def main():
    print("Housing & Development Bank RAG Bot")

    collection_info = get_collection_info()

    if collection_info["count"] == 0:
        print("Vector database is empty.")
        print("Indexing documents...")

        generate_and_index_embeddings()

        print("Documents indexed successfully.")
    else:
        print(
            f"Vector database already contains "
            f"{collection_info['count']} documents."
        )

    print("\nType 'exit' or 'quit' to stop.")

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        answer = generate_answer(question)

        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()