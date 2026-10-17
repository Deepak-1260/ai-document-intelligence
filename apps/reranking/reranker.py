
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


# Load the pretrained cross-encoder once when this module is imported.
# The model is downloaded from Hugging Face on its first use and cached locally.
# It does not require Ollama or a hosted reranking API.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

reranker = CrossEncoder(
    MODEL_NAME,
    device="cpu",  # Use CPU for a straightforward local setup.
)


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 3,
) -> List[Tuple[Document, float]]:
    """
    Reorder retrieved LangChain Documents by their relevance to the query.

    Args:
        query: The original question asked by the user.
        documents: Candidate documents from Chroma or Multi-Query Retrieval.
        top_n: Maximum number of top-ranked documents to return.

    Returns:
        A list of (Document, score) tuples, sorted from highest score
        to lowest score.
    """

    # Step 1: Handle empty retrieval results.
    # There is nothing to score if the retriever returned no documents.
    if not documents:
        return []

    # Step 2: Validate the requested number of results.
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    # Step 3: Create query-document pairs.
    # A cross-encoder processes the question and each document TOGETHER.
    # Unlike vector search, it does not compare precomputed embeddings.
    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    # Step 4: Score every candidate document against the original question.
    # predict() returns one relevance score for each input pair.
    scores = reranker.predict(pairs)

    # Step 5: Associate each score with its original Document.
    # Keeping the Document object preserves its page content and metadata.
    scored_documents = list(zip(documents, scores))

    # Step 6: Sort by score in descending order.
    # Higher scores indicate stronger relevance according to this model.
    ranked_documents = sorted(
        scored_documents,
        key=lambda item: float(item[1]),
        reverse=True,
    )

    # Step 7: Return only the requested number of top-ranked documents.
    # Convert model scores to ordinary Python floats for convenient use.
    return [
        (document, float(score))
        for document, score in ranked_documents[:top_n]
    ]


