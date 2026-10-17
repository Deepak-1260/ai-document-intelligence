
import csv
import json
import re
from pathlib import Path
from statistics import mean

# Project paths
ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = ROOT / "data" / "evaluation"

GROUND_TRUTH_PATH = EVALUATION_DIR / "question_relevant_chunk_ids.csv"
CHUNKS_PATH = EVALUATION_DIR / "unique_chunks_id_and_text.csv"
CANDIDATES_PATH = EVALUATION_DIR / "exhaustive_candidates.json"

K_VALUES = (5, 10, 20)


def normalize_text(text: str) -> str:
    """Normalize whitespace so identical chunks can be matched reliably."""
    return re.sub(r"\s+", " ", text).strip()


def load_ground_truth():
    """Load relevant chunk IDs for every question."""
    ground_truth = {}

    with open(GROUND_TRUTH_PATH, encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            # Relevant chunk IDs are separated by semicolons.
            relevant_ids = {
                chunk_id.strip()
                for chunk_id in row["relevant_chunk_ids"].split(";")
                if chunk_id.strip()
            }

            ground_truth[row["question_id"]] = {
                "relevant_chunk_ids": relevant_ids,
                "answerable": row["answerable"].strip().lower() == "true",
            }

    return ground_truth


def load_chunk_id_mapping():
    """Build a mapping from normalized chunk text to its stable ID."""
    text_to_id = {}

    with open(CHUNKS_PATH, encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            normalized = normalize_text(row["chunk"])
            chunk_id = row["chunk_id"].strip()

            if normalized in text_to_id:
                raise ValueError(
                    f"Duplicate chunk text found for {chunk_id}"
                )

            text_to_id[normalized] = chunk_id

    return text_to_id


def calculate_metrics(ranked_chunk_ids, relevant_chunk_ids):
    """Calculate Precision@K, Recall@K and Reciprocal Rank."""
    labels = [
        1 if chunk_id in relevant_chunk_ids else 0
        for chunk_id in ranked_chunk_ids
    ]

    # Rank of the first relevant chunk, if one was retrieved.
    first_relevant_rank = next(
        (rank for rank, label in enumerate(labels, start=1) if label == 1),
        None,
    )

    metrics = {
        "mrr": 1 / first_relevant_rank if first_relevant_rank else 0.0
    }

    for k in K_VALUES:
        top_k_labels = labels[:k]

        # Precision@K uses K as the denominator.
        metrics[f"precision@{k}"] = sum(top_k_labels) / k

        # Recall is undefined when no relevant chunks are labeled.
        metrics[f"recall@{k}"] = (
            sum(top_k_labels) / len(relevant_chunk_ids)
            if relevant_chunk_ids
            else None
        )

    return metrics


def main():
    ground_truth = load_ground_truth()
    text_to_id = load_chunk_id_mapping()

    with open(CANDIDATES_PATH, encoding="utf-8") as file:
        exhaustive = json.load(file)

    candidates_by_question = {
        case["id"]: case["candidates"]
        for case in exhaustive
    }

    # Validate that both datasets contain the same test cases.
    if set(ground_truth) != set(candidates_by_question):
        raise ValueError(
            "Test-case IDs do not match between ground truth "
            "and exhaustive candidates."
        )

    per_case = []

    for question_id, judgment in ground_truth.items():
        candidates = candidates_by_question[question_id]

        # Convert every ranked candidate's text into its stable chunk ID.
        ranked_chunk_ids = []

        for candidate in candidates:
            normalized = normalize_text(candidate["content"])

            if normalized not in text_to_id:
                raise ValueError(
                    f"Could not map {question_id}, rank "
                    f"{candidate['rank']} to a chunk ID."
                )

            ranked_chunk_ids.append(text_to_id[normalized])

        # Every labeled relevant ID must exist in the chunk collection.
        missing_ids = (
            judgment["relevant_chunk_ids"] - set(text_to_id.values())
        )
        if missing_ids:
            raise ValueError(
                f"{question_id} has unknown relevant chunk IDs: {missing_ids}"
            )

        metrics = calculate_metrics(
            ranked_chunk_ids,
            judgment["relevant_chunk_ids"],
        )

        per_case.append({
            "id": question_id,
            "answerable": judgment["answerable"],
            "num_relevant_chunks": len(judgment["relevant_chunk_ids"]),
            **metrics,
        })

    # Display metrics for every test case.
    print("\nPER-TEST-CASE METRICS")

    for row in per_case:
        formatted = []

        for key, value in row.items():
            if key in ("id", "answerable", "num_relevant_chunks"):
                continue

            formatted.append(
                f"{key}={value:.3f}"
                if value is not None
                else f"{key}=N/A"
            )

        print(f"{row['id']}: " + ", ".join(formatted))

    # Aggregate answerable questions with at least one relevant chunk.
    evaluable = [
        row for row in per_case
        if row["answerable"] and row["num_relevant_chunks"] > 0
    ]

    print(
        "\nMACRO-AVERAGE METRICS "
        "(answerable questions with relevant chunk IDs)"
    )

    if not evaluable:
        print("No evaluable questions found.")
        return

    metric_names = (
        [f"precision@{k}" for k in K_VALUES]
        + [f"recall@{k}" for k in K_VALUES]
        + ["mrr"]
    )

    for metric in metric_names:
        values = [
            row[metric]
            for row in evaluable
            if row[metric] is not None
        ]
        print(f"{metric}: {mean(values):.3f}")


if __name__ == "__main__":
    main()
