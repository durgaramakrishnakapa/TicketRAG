"""
Query ChromaDB only (no indexing). Use after index_data.py has been run once.
Queries all 6 DBs in parallel, top 3 per chunk; answers with Gemini. All LLM code here.
Supports hybrid retrieval: dense (ChromaDB) + sparse (BM25).
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai
from google.genai import types

from bm25_index import query_bm25_split

load_dotenv()

DATA_DIR = "data"
DB_DIR = os.path.join(DATA_DIR, "db")
N_SPLITS = 6
DB_PREFIX = "chroma_db_part"
COLLECTION_NAME = "tickets"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_PER_CHUNK = 6
GEMINI_MODEL = "gemini-2.5-flash"

EF = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _gemini_generate(
    prompt: str,
    *,
    model: str = GEMINI_MODEL,
    system_instruction: str | None = None,
    temperature: float = 0.3,
    api_key: str | None = None,
) -> str:
    """Call Gemini LLM and return generated text. All LLM code in this file."""
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("No Gemini API key: set GEMINI_API_KEY or pass gemini_api_key.")
    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
    )
    response = client.models.generate_content(model=model, contents=prompt, config=config)
    return response.text or ""


def rerank_with_cross_encoder(
    query: str,
    documents: list,
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list:
    """
    Rerank documents using cross-encoder model.
    Returns top_k most relevant documents based on cross-encoder scoring.
    
    Args:
        query: Query string
        documents: List of document dictionaries
        top_k: Number of top documents to return
        model_name: Cross-encoder model name
    """
    if not documents:
        return []
    
    try:
        from sentence_transformers import CrossEncoder
        
        # Load cross-encoder model
        model = CrossEncoder(model_name)
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            doc_text = doc.get("document", "")
            pairs.append([query, doc_text])
        
        # Score all pairs
        scores = model.predict(pairs)
        
        # Attach scores to documents
        scored_docs = []
        for i, doc in enumerate(documents):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(scores[i])
            scored_docs.append(doc_copy)
        
        # Sort by rerank score (higher is better)
        scored_docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return scored_docs[:top_k]
    
    except Exception as e:
        print(f"Warning: Cross-encoder reranking failed ({e}), returning original order")
        return documents[:top_k]


def _query_one_chunk(args):
    """Query a single ChromaDB; returns list of results for that chunk."""
    chunk_id, query_text, n_per_chunk = args
    path = os.path.join(DB_DIR, f"{DB_PREFIX}_{chunk_id}")
    if not os.path.isdir(path):
        return []
    client = chromadb.PersistentClient(path=path)
    coll = client.get_collection(COLLECTION_NAME, embedding_function=EF)
    r = coll.query(query_texts=[query_text], n_results=n_per_chunk)
    chunk_results = []
    for j, doc in enumerate(r["documents"][0]):
        chunk_results.append({
            "document": doc,
            "metadata": r["metadatas"][0][j] if r["metadatas"] else {},
            "distance": r["distances"][0][j] if r.get("distances") else None,
            "chunk_id": chunk_id,
        })
    return chunk_results


def query_all_splits(query_text: str, n_per_chunk: int = TOP_K_PER_CHUNK, max_total: int | None = None):
    """Query all ChromaDB splits in parallel; top n_per_chunk from each, merge by distance."""
    chunk_ids = [i for i in range(1, N_SPLITS + 1) if os.path.isdir(os.path.join(DB_DIR, f"{DB_PREFIX}_{i}"))]
    if not chunk_ids:
        return []

    all_results = []
    with ThreadPoolExecutor(max_workers=len(chunk_ids)) as executor:
        futures = {
            executor.submit(_query_one_chunk, (i, query_text, n_per_chunk)): i
            for i in chunk_ids
        }
        for future in as_completed(futures):
            chunk_results = future.result()
            all_results.extend(chunk_results)

    all_results.sort(key=lambda x: x["distance"] or float("inf"))
    if max_total is not None:
        all_results = all_results[:max_total]
    return all_results


def query_bm25_all_splits(query_text: str, n_per_chunk: int = TOP_K_PER_CHUNK):
    """Query all BM25 splits in parallel; top n_per_chunk from each."""
    chunk_ids = list(range(1, N_SPLITS + 1))
    
    all_results = []
    with ThreadPoolExecutor(max_workers=len(chunk_ids)) as executor:
        futures = {
            executor.submit(query_bm25_split, i, query_text, n_per_chunk): i
            for i in chunk_ids
        }
        for future in as_completed(futures):
            chunk_results = future.result()
            all_results.extend(chunk_results)
    
    # Sort by BM25 score (higher is better)
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_results


def normalize_score(score: float, min_score: float, max_score: float) -> float:
    """Normalize score to 0-1 range."""
    if max_score == min_score:
        return 0.5
    return (score - min_score) / (max_score - min_score)


def hybrid_query(
    query_text: str,
    n_per_chunk: int = TOP_K_PER_CHUNK,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    max_total: int | None = None,
):
    """
    Hybrid retrieval: combine dense (ChromaDB) and sparse (BM25) results.
    
    Args:
        query_text: Query string
        n_per_chunk: Top k results per chunk for each method
        dense_weight: Weight for dense retrieval (0-1)
        sparse_weight: Weight for sparse retrieval (0-1)
        max_total: Maximum total results to return
    
    Returns:
        List of results sorted by combined score
    """
    # Get dense results
    dense_results = query_all_splits(query_text, n_per_chunk=n_per_chunk, max_total=None)
    
    # Get sparse results
    sparse_results = query_bm25_all_splits(query_text, n_per_chunk=n_per_chunk)
    
    # Normalize scores
    if dense_results:
        dense_distances = [r.get("distance", 0) for r in dense_results]
        min_dist, max_dist = min(dense_distances), max(dense_distances)
        for r in dense_results:
            # Convert distance to similarity (lower distance = higher similarity)
            r["normalized_score"] = 1 - normalize_score(r.get("distance", 0), min_dist, max_dist)
    
    if sparse_results:
        sparse_scores = [r.get("score", 0) for r in sparse_results]
        min_score, max_score = min(sparse_scores), max(sparse_scores)
        for r in sparse_results:
            r["normalized_score"] = normalize_score(r.get("score", 0), min_score, max_score)
    
    # Combine results by doc_id
    combined = {}
    
    for r in dense_results:
        doc_id = r.get("chunk_id", "")
        combined[doc_id] = {
            **r,
            "dense_score": r.get("normalized_score", 0),
            "sparse_score": 0,
            "retrieval_method": "dense",
        }
    
    for r in sparse_results:
        doc_id = r.get("doc_id", "")
        if doc_id in combined:
            combined[doc_id]["sparse_score"] = r.get("normalized_score", 0)
            combined[doc_id]["retrieval_method"] = "hybrid"
        else:
            combined[doc_id] = {
                **r,
                "dense_score": 0,
                "sparse_score": r.get("normalized_score", 0),
                "retrieval_method": "sparse",
            }
    
    # Calculate final combined score
    results = []
    for doc_id, r in combined.items():
        r["combined_score"] = (
            dense_weight * r["dense_score"] + 
            sparse_weight * r["sparse_score"]
        )
        results.append(r)
    
    # Sort by combined score
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    
    if max_total is not None:
        results = results[:max_total]
    
    return results


def print_chunk_results(results: list, doc_preview_len: int = 400):
    """Print each retrieved chunk: chunk_id, distance, metadata, document preview."""
    print(f"\n{'='*60}\nRetrieved {len(results)} chunks (top 3 per DB × 6 DBs)\n{'='*60}")
    for i, r in enumerate(results, 1):
        dist = r.get("distance")
        dist_str = f"{dist:.4f}" if dist is not None else "N/A"
        meta = r.get("metadata") or {}
        meta_str = ", ".join(f"{k}={v}" for k, v in list(meta.items())[:5])
        doc = (r.get("document") or "")[:doc_preview_len]
        if len(r.get("document") or "") > doc_preview_len:
            doc += "..."
        print(f"\n--- Chunk {i} (DB part {r.get('chunk_id', '?')}, distance={dist_str}) ---")
        if meta_str:
            print(f"  Metadata: {meta_str}")
        print(f"  Document preview:\n  {doc.replace(chr(10), chr(10) + '  ')}")
    print(f"\n{'='*60}\n")


def answer_with_gemini(
    query: str,
    context_docs: list,
    model: str = GEMINI_MODEL,
    api_key: str | None = None,
    use_reranking: bool = False,
    use_citations: bool = True,
) -> dict:
    """
    Build RAG prompt and get answer from Gemini with citations. All LLM code in this file.
    
    Args:
        query: User query
        context_docs: Retrieved documents
        model: Gemini model name
        api_key: API key
        use_reranking: If True, rerank documents with cross-encoder before answering
        use_citations: If True, include citations in answer
    
    Returns:
        dict with 'answer' and 'sources' keys
    """
    # Optionally rerank documents with cross-encoder
    if use_reranking:
        context_docs = rerank_with_cross_encoder(query, context_docs, top_k=5)
    
    # Build context with citations
    context_parts = []
    sources = []
    
    for i, doc in enumerate(context_docs, 1):
        metadata = doc.get("metadata", {})
        doc_text = doc.get("document", "")
        
        # Extract key metadata
        subject = metadata.get("subject", "N/A")
        ticket_type = metadata.get("type", "N/A")
        priority = metadata.get("priority", "N/A")
        queue = metadata.get("queue", "N/A")
        
        # Build citation reference
        if use_citations:
            context_parts.append(f"[{i}] {doc_text}")
            sources.append({
                "citation_id": i,
                "subject": subject,
                "type": ticket_type,
                "priority": priority,
                "queue": queue,
                "chunk_id": doc.get("chunk_id", "N/A"),
            })
        else:
            context_parts.append(doc_text)
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Build system instruction with citation guidance
    if use_citations:
        system = (
            "You are a helpful support assistant. Answer based only on the following ticket context. "
            "Each document is numbered with [1], [2], etc. "
            "When you use information from a document, cite it using the number like [1] or [2]. "
            "You can cite multiple sources like [1][2]. "
            "If the context does not contain enough information, say so briefly."
        )
    else:
        system = (
            "You are a helpful support assistant. Answer based only on the following ticket context. "
            "If the context does not contain enough information, say so briefly."
        )
    
    prompt = f"Context from knowledge base:\n\n{context}\n\nQuestion: {query}"
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    answer_text = _gemini_generate(
        prompt,
        model=model,
        system_instruction=system,
        temperature=0.3,
        api_key=key,
    )
    
    return {
        "answer": answer_text,
        "sources": sources if use_citations else [],
    }


def main():
    import sys
    argv = sys.argv[1:]
    chunks_only = "--chunks-only" in argv or "-c" in argv
    use_hybrid = "--hybrid" in argv or "-h" in argv
    use_reranking = "--rerank" in argv or "-r" in argv
    
    if chunks_only:
        argv = [a for a in argv if a not in ("--chunks-only", "-c")]
    if use_hybrid:
        argv = [a for a in argv if a not in ("--hybrid", "-h")]
    if use_reranking:
        argv = [a for a in argv if a not in ("--rerank", "-r")]
    
    if not chunks_only and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment.")
        sys.exit(1)
    
    query = " ".join(argv) if argv else "How do I fix VPN connection issues?"
    print("Query:", query)
    
    if use_hybrid:
        print("\nQuerying with HYBRID retrieval (dense + sparse BM25)...")
        results = hybrid_query(query, n_per_chunk=TOP_K_PER_CHUNK, max_total=36)
        print(f"\nRetrieved {len(results)} results using hybrid search")
        for i, r in enumerate(results[:10], 1):
            method = r.get("retrieval_method", "unknown")
            combined = r.get("combined_score", 0)
            dense = r.get("dense_score", 0)
            sparse = r.get("sparse_score", 0)
            print(f"{i}. [{method}] Combined={combined:.3f} (dense={dense:.3f}, sparse={sparse:.3f})")
    else:
        print("\nQuerying ChromaDB (dense only, parallel, top 6 per chunk)...")
        results = query_all_splits(query, n_per_chunk=TOP_K_PER_CHUNK, max_total=None)
    
    if not results:
        print("No results. Run index_data.py first to create data/db/chroma_db_part_1..6.")
        if use_hybrid:
            print("Also run: python bm25_index.py to create BM25 indices.")
        sys.exit(1)
    
    print_chunk_results(results)
    
    if chunks_only:
        print("(Chunks only; omit --chunks-only to get Gemini answer.)")
        return
    
    if use_reranking:
        print("\nReranking with cross-encoder (top 5)...")
    
    print("Asking Gemini...")
    result = answer_with_gemini(query, results, use_reranking=use_reranking, use_citations=True)
    
    print("\nAnswer:\n", result["answer"])
    
    if result.get("sources"):
        print("\n" + "="*60)
        print("SOURCES:")
        print("="*60)
        for src in result["sources"]:
            print(f"[{src['citation_id']}] {src['subject']}")
            print(f"    Type: {src['type']} | Priority: {src['priority']} | Queue: {src['queue']}")
            print()


if __name__ == "__main__":
    main()
