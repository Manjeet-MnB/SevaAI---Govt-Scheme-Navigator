"""
Step 1: Embed all schemes and save FAISS index.
Run this once before starting the app.

Usage:
    pip install openai faiss-cpu numpy
    export OPENAI_API_KEY=your_key_here
    python embed_schemes.py
"""

import json
import os
import pickle
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SCHEMES_PATH = "data/schemes.json"
INDEX_PATH   = "embeddings/faiss_index.pkl"
META_PATH    = "embeddings/scheme_meta.pkl"
EMBED_MODEL  = "text-embedding-3-small"   # cheapest + good quality


def scheme_to_text(scheme: dict) -> str:
    """Convert a scheme dict into a rich text blob for embedding."""
    elig = scheme["eligibility"]
    caste_str = ", ".join(elig.get("caste", []))
    occ_str   = ", ".join(elig.get("occupation", []))
    steps_str = " | ".join(scheme.get("how_to_apply", []))
    return (
        f"Scheme: {scheme['name']}. "
        f"Ministry: {scheme['ministry']}. "
        f"Category: {', '.join(scheme.get('category', []))}. "
        f"Eligible occupation: {occ_str}. "
        f"Eligible caste: {caste_str}. "
        f"Income criteria: {elig.get('income') or 'no restriction'}. "
        f"Age: {elig.get('age_min', 0)} to {elig.get('age_max') or 'any'}. "
        f"Benefits: {scheme['benefits']}. "
        f"Application steps: {steps_str}. "
        f"Tags: {scheme.get('tags', '')}."
    )


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Batch embed texts using OpenAI. Returns (N, 1536) float32 array."""
    # OpenAI allows up to 2048 inputs per batch
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    vecs = [item.embedding for item in response.data]
    return np.array(vecs, dtype="float32")


def build_index():
    try:
        import faiss
    except ImportError:
        print("Installing faiss-cpu...")
        os.system("pip install faiss-cpu --quiet --break-system-packages")
        import faiss

    with open(SCHEMES_PATH) as f:
        schemes = json.load(f)

    print(f"Loaded {len(schemes)} schemes.")

    texts = [scheme_to_text(s) for s in schemes]
    print("Generating embeddings (using text-embedding-3-small)...")
    embeddings = get_embeddings(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)   # Inner product (cosine if normalized)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    os.makedirs("embeddings", exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(faiss.serialize_index(index), f)

    meta = [{"id": s["id"], "name": s["name"], "scheme": s} for s in schemes]
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"Index saved: {len(schemes)} schemes embedded ({dimension}D).")
    print(f"  {INDEX_PATH}")
    print(f"  {META_PATH}")


if __name__ == "__main__":
    build_index()
