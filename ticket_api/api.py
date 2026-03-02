"""
FastAPI: query parameters api_key (optional), ticket (required).
Uses query_data.py for ChromaDB + Gemini.
"""
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from query_data import query_all_splits, answer_with_gemini, TOP_K_PER_CHUNK

load_dotenv()

app = FastAPI(title="Ticket RAG API", version="1.0.0")


@app.get("/query")
def query_tickets(
    ticket: str = Query(..., min_length=1, description="Ticket question"),
    api_key: Optional[str] = Query(None, description="Gemini API key (optional; uses .env if not set)"),
    use_hybrid: bool = Query(False, description="Use hybrid retrieval (dense + BM25)"),
    use_reranking: bool = Query(False, description="Use cross-encoder reranking (top 5 most relevant)"),
    use_citations: bool = Query(True, description="Include citations and source metadata in answer"),
):
    """
    Query ticket knowledge base and get Gemini answer with citations.
    Query params: ticket (required), api_key (optional), use_hybrid (optional), use_reranking (optional), use_citations (optional).
    """
    ticket = ticket.strip()
    if not ticket:
        raise HTTPException(status_code=400, detail="ticket must be non-empty")

    if use_hybrid:
        from query_data import hybrid_query
        chunks_raw = hybrid_query(ticket, n_per_chunk=TOP_K_PER_CHUNK, max_total=36)
    else:
        chunks_raw = query_all_splits(ticket, n_per_chunk=TOP_K_PER_CHUNK, max_total=None)
    
    if not chunks_raw:
        raise HTTPException(status_code=503, detail="No results. Run index_data.py first (data/db/).")

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    answer_result = None
    if key:
        try:
            answer_result = answer_with_gemini(
                ticket, 
                chunks_raw, 
                api_key=key, 
                use_reranking=use_reranking,
                use_citations=use_citations
            )
        except Exception as e:
            err = str(e)
            status = 400 if "API key" in err and ("invalid" in err.lower() or "INVALID" in err) else 502
            raise HTTPException(status_code=status, detail=f"Gemini: {err}")

    chunks_out = [
        {"chunk_id": r["chunk_id"], "distance": r.get("distance"), "metadata": r.get("metadata") or {}, "document": (r.get("document") or "")[:500]}
        for r in chunks_raw
    ]
    
    response = {
        "ticket": ticket, 
        "chunks": chunks_out, 
        "retrieval_method": "hybrid" if use_hybrid else "dense",
        "reranking_enabled": use_reranking,
        "citations_enabled": use_citations,
    }
    
    if answer_result:
        response["answer"] = answer_result.get("answer")
        response["sources"] = answer_result.get("sources", [])
    else:
        response["answer"] = None
        response["sources"] = []
    
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
