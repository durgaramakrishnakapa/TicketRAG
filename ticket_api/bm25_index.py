"""
BM25 sparse retrieval index for tickets.
Complements dense retrieval with keyword-based search.
"""
import os
import pickle
from typing import List, Dict, Any

import pandas as pd
from rank_bm25 import BM25Okapi


DATA_DIR = "data"
BM25_DIR = os.path.join(DATA_DIR, "bm25")
N_SPLITS = 6
CSV_PREFIX = "aa_dataset-tickets-en-only"


def tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase and split on whitespace."""
    return text.lower().split()


def build_document(row) -> str:
    """Build searchable document from ticket row."""
    parts = [
        str(row.get("subject", "") or ""),
        str(row.get("body", "") or ""),
        str(row.get("answer", "") or ""),
    ]
    return " ".join(p for p in parts if p.strip())


def index_bm25_split(part_num: int):
    """Create BM25 index for one CSV split."""
    csv_path = os.path.join(DATA_DIR, f"{CSV_PREFIX}-part-{part_num}-of-{N_SPLITS}.csv")
    
    if not os.path.isfile(csv_path):
        print(f"  Skip part {part_num}: {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    # Build documents and tokenize
    documents = []
    doc_ids = []
    metadatas = []
    
    for idx, row in df.iterrows():
        doc = build_document(row)
        documents.append(doc)
        doc_ids.append(f"part{part_num}_row{idx}")
        metadatas.append(row.to_dict())
    
    # Tokenize all documents
    tokenized_docs = [tokenize(doc) for doc in documents]
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_docs)
    
    # Save index, documents, and metadata
    os.makedirs(BM25_DIR, exist_ok=True)
    index_path = os.path.join(BM25_DIR, f"bm25_part_{part_num}.pkl")
    
    with open(index_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "documents": documents,
            "doc_ids": doc_ids,
            "metadatas": metadatas,
        }, f)
    
    print(f"  Part {part_num}: {len(documents):,} documents -> {index_path}")


def load_bm25_split(part_num: int) -> Dict[str, Any]:
    """Load BM25 index for one split."""
    index_path = os.path.join(BM25_DIR, f"bm25_part_{part_num}.pkl")
    
    if not os.path.isfile(index_path):
        return None
    
    with open(index_path, "rb") as f:
        return pickle.load(f)


def query_bm25_split(part_num: int, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Query one BM25 split and return top results."""
    data = load_bm25_split(part_num)
    if not data:
        return []
    
    bm25 = data["bm25"]
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    # Get top k indices
    top_indices = scores.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "document": data["documents"][idx],
            "doc_id": data["doc_ids"][idx],
            "metadata": data["metadatas"][idx],
            "score": float(scores[idx]),
            "chunk_id": part_num,
        })
    
    return results


def main():
    """Index all splits with BM25."""
    print("Creating BM25 indices for all splits...")
    for i in range(1, N_SPLITS + 1):
        index_bm25_split(i)
    print("Done. BM25 indices saved to data/bm25/")


if __name__ == "__main__":
    main()
