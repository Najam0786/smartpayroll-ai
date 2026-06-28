# src/rag/document_processor.py
"""
Document processing pipeline for HR policy RAG.
Chunks documents, generates embeddings, stores in Azure AI Search.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()
logger = logging.getLogger(__name__)


def get_openai_client() -> AzureOpenAI:
    """
    Create Azure OpenAI client using API key.
    """
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_PROJECT_ENDPOINT"],
        api_key=os.environ["AZURE_API_KEY"],
        api_version="2024-02-01",
    )


def chunk_document(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Split document text into overlapping chunks.

    Why chunks?
    Embedding models have token limits.
    Smaller chunks = more precise retrieval.
    Overlap ensures context is not lost at boundaries.

    Args:
        text: Full document text
        chunk_size: Characters per chunk
        chunk_overlap: Characters of overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)

        if end == len(text):
            break

        start += chunk_size - chunk_overlap

    logger.info(f"Document split into {len(chunks)} chunks")
    return chunks


def embed_texts(
    texts: list[str],
    client: AzureOpenAI
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Embedding = converting text to a vector of numbers.
    Similar texts have similar vectors.
    This enables semantic search.

    Args:
        texts: List of text strings to embed
        client: Azure OpenAI client

    Returns:
        List of embedding vectors
    """
    response = client.embeddings.create(
        input=texts,
        model=os.environ.get(
            "AZURE_EMBEDDING_DEPLOYMENT",
            "text-embedding-3-small"
        ),
    )

    embeddings = [item.embedding for item in response.data]
    logger.info(
        f"Generated {len(embeddings)} embeddings, "
        f"dimension={len(embeddings[0])}"
    )
    return embeddings


def process_policy_documents(
    policies_dir: str = "hr_policies",
) -> list[dict]:
    """
    Process all HR policy documents.
    Chunks each document and generates embeddings.

    Args:
        policies_dir: Directory containing .md policy files

    Returns:
        List of dicts with content, embedding, metadata
    """
    client = get_openai_client()
    policy_files = list(Path(policies_dir).glob("*.md"))

    logger.info(f"Found {len(policy_files)} policy documents")

    all_chunks = []

    for file_path in policy_files:
        # Read document
        content = file_path.read_text(encoding="utf-8")
        filename = file_path.stem

        # Extract country from filename
        parts = filename.split("_")
        country = parts[-1] if len(parts[-1]) == 2 else "ALL"

        logger.info(f"Processing: {filename} ({len(content)} chars)")

        # Chunk the document
        chunks = chunk_document(content)

        # Generate embeddings for all chunks at once
        embeddings = embed_texts(chunks, client)

        # Build document records
        for i, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            all_chunks.append({
                "id": f"{filename}_chunk_{i}",
                "content": chunk,
                "source": filename,
                "country": country,
                "chunk_index": i,
                "embedding": embedding,
            })

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Processing HR policy documents...")
    chunks = process_policy_documents()

    print(f"\n{'='*50}")
    print("DOCUMENT PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Total chunks: {len(chunks)}")
    print("\nSample chunk:")
    print(f"  ID:      {chunks[0]['id']}")
    print(f"  Source:  {chunks[0]['source']}")
    print(f"  Country: {chunks[0]['country']}")
    print(f"  Length:  {len(chunks[0]['content'])} chars")
    print(f"  Embedding dim: {len(chunks[0]['embedding'])}")