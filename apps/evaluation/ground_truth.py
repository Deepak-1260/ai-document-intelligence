import json
from pathlib import Path

from embeddings.embedder import embedder
from vector_store.store import vector_store


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "rag_test_cases.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "exhaustive_candidates.json"


def find_ground_truth_candidates():
    # Load evaluation questions and expected answers
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    # Open the existing persistent Chroma vector store
    store = vector_store(embedder)

    # Get the number of chunks currently stored
    total_chunks = store._collection.count()

    results = []

    for test_case in test_cases:
        question = test_case["question"]

        # Retrieve every stored chunk, ordered by vector distance
        retrieved = store.similarity_search_with_score(
            query=question,
            k=total_chunks
        )

        candidates = []

        for rank, (document, distance) in enumerate(retrieved, start=1):
            candidates.append({
                "rank": rank,
                "distance": float(distance),
                "content": document.page_content,
                "metadata": document.metadata
            })

        results.append({
            "id": test_case["id"],
            "question": question,
            "expected_answer": test_case["expected_answer"],
            "answerable": test_case["answerable"],
            "relevant_pages": test_case["relevant_pages"],
            "total_candidates": len(candidates),
            "candidates": candidates
        })

        print(
            f"Processed {test_case['id']}: "
            f"{len(candidates)} candidate chunks"
        )

    # Save candidate results for inspection
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nTotal chunks in vector store: {total_chunks}")
    print(f"Candidate results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    find_ground_truth_candidates()