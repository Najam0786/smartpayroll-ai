# src/rag/chain.py
"""
RAG chain: search + generate.
Answers HR policy questions using retrieved documents.
"""

import logging
import os
import time

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert HR policy specialist for SD Worx.
Answer questions about HR policies accurately and concisely.

STRICT RULES:
1. Answer ONLY based on the provided policy documents
2. Always cite the source document
3. If answer is not in documents, say so clearly
4. End every response with:
   "Note: Verify with your HR team before taking action."

POLICY DOCUMENTS:
{context}"""


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Calculate cosine similarity between two vectors.

    1.0  = identical (very similar)
    0.0  = unrelated
    -1.0 = opposite

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Similarity score between -1 and 1
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def retrieve_relevant_chunks(
    query: str,
    chunks: list[dict],
    client: AzureOpenAI,
    top_k: int = 3,
) -> list[dict]:
    """
    Find most relevant chunks for a query using cosine similarity.

    Args:
        query: User question
        chunks: All processed document chunks with embeddings
        client: Azure OpenAI client
        top_k: Number of chunks to return

    Returns:
        Top-k most relevant chunks
    """
    # Embed the query
    response = client.embeddings.create(
        input=[query],
        model=os.environ.get(
            "AZURE_EMBEDDING_DEPLOYMENT",
            "text-embedding-3-small"
        ),
    )
    query_embedding = response.data[0].embedding

    # Calculate similarity with all chunks
    scored_chunks = []
    for chunk in chunks:
        score = cosine_similarity(
            query_embedding,
            chunk["embedding"]
        )
        scored_chunks.append({**chunk, "score": score})

    # Sort by similarity and return top-k
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = scored_chunks[:top_k]

    logger.info(
        f"Top chunk scores: "
        f"{[round(c['score'], 3) for c in top_chunks]}"
    )

    return top_chunks


def answer_question(
    question: str,
    chunks: list[dict],
) -> dict:
    """
    Answer an HR policy question using RAG.

    Steps:
    1. Embed the question
    2. Find most similar chunks (retrieval)
    3. Build context from retrieved chunks
    4. Generate answer with Phi-4-mini-instruct

    Args:
        question: HR policy question
        chunks: All processed document chunks

    Returns:
        dict with answer, sources, and scores
    """
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_PROJECT_ENDPOINT"],
        api_key=os.environ["AZURE_API_KEY"],
        api_version="2024-02-01",
    )

    # Step 1 & 2: Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(
        question, chunks, client, top_k=3
    )

    # Step 3: Build context
    context = "\n\n".join([
        f"[Source: {c['source']}]\n{c['content']}"
        for c in relevant_chunks
    ])
    sources = list(set(c["source"] for c in relevant_chunks))

    # Step 4: Generate answer
    response = client.chat.completions.create(
        model=os.environ.get(
            "AZURE_CHAT_DEPLOYMENT",
            "Phi-4-mini-instruct"
        ),
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(context=context)
            },
            {
                "role": "user",
                "content": question
            }
        ],
        max_tokens=500,
        temperature=0.0,
    )

    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "chunks_used": len(relevant_chunks),
        "top_score": round(relevant_chunks[0]["score"], 3),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.rag.document_processor import process_policy_documents

    # Process documents
    print("Loading and processing documents...")
    chunks = process_policy_documents()

    # Test ONE question to avoid rate limit
    test_questions = [
        "What is the annual leave entitlement in Spain?",
        "How is overtime compensated at SD Worx?",
        "What is the notice period for 5 days of leave?",
    ]

    print(f"\n{'='*60}")
    print("RAG PIPELINE — TEST QUERIES")
    print("=" * 60)

    for question in test_questions:
        print(f"\nQ: {question}")

        try:
            result = answer_question(question, chunks)
            print(f"A: {result['answer']}")
            print(f"Sources: {result['sources']}")
            print(f"Score:   {result['top_score']}")

        except Exception as e:
            print(f"Error: {e}")

        print("-" * 60)

        # Wait between calls to avoid rate limit
        time.sleep(10)