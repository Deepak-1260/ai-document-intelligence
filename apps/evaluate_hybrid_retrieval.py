
import csv
import json
import re
from pathlib import Path

# Import your existing hybrid retriever.
# If its variable has a different name, update this import.
from retriever.hybrid_retriever import hybrid_search


# Project root: ai-document-intelligence/
ROOT = Path(__file__).resolve().parents[1]

TEST_CASES_PATH = ROOT / "data/evaluation/rag_test_cases.json"
CHUNKS_PATH = ROOT / "data/evaluation/unique_chunks_id_and_text.csv"
OUTPUT_PATH = ROOT / "data/evaluation/hybrid_results.json"

# Retrieve the top 20 chunks for every question.
TOP_K = 20


def normalize_text(text: str) -> str:
    """Normalize whitespace so equivalent chunk text can be matched."""
    return re.sub(r"\s+", " ", text).strip()


def load_chunk_id_mapping():
    """Create a mapping from normalized chunk text to its stable ID."""
    text_to_id = {}

    with CHUNKS_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            chunk_id = row["chunk_id"]
            chunk_text = normalize_text(row["chunk"])

            # Stop if identical text has conflicting IDs.
            if chunk_text in text_to_id and text_to_id[chunk_text] != chunk_id:
                raise ValueError(
                    f"Conflicting chunk IDs for identical text: {chunk_text[:80]}"
                )

            text_to_id[chunk_text] = chunk_id

    return text_to_id


def main():
    # Load the 24 evaluation questions.
    with TEST_CASES_PATH.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    # Convert chunk text to stable chunk IDs.
    text_to_id = load_chunk_id_mapping()

    results = []

    for case in test_cases:
        question_id = case["id"]
        question = case["question"]

        print(f"Retrieving: {question_id}")

        # Invoke the existing hybrid retriever.
        documents = hybrid_search(question)

        ranked_chunks = []
        seen_chunk_ids = set()

        for document in documents:
            # Map the retrieved document to its stable chunk ID.
            normalized_text = normalize_text(document.page_content)
            chunk_id = text_to_id.get(normalized_text)

            if chunk_id is None:
                raise ValueError(
                    f"{question_id}: Could not map retrieved document to "
                    f"a stable chunk ID. Text: {normalized_text[:120]}"
                )

            # Avoid counting the same chunk multiple times in one ranking.
            if chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk_id)

            ranked_chunks.append({
                "rank": len(ranked_chunks) + 1,
                "chunk_id": chunk_id
            })

            if len(ranked_chunks) == TOP_K:
                break

        results.append({
            "question_id": question_id,
            "question": question,
            "retrieved_chunks": ranked_chunks
        })

        print(f"  Retrieved {len(ranked_chunks)} unique chunks")

    # Save the ranked results for the evaluation step.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nSaved hybrid retrieval results to: {OUTPUT_PATH}")
    print(f"Questions processed: {len(results)}")


if __name__ == "__main__":
    main()
