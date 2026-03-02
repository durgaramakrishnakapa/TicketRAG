# Hybrid Retrieval with Citation Grounding

This project now supports **hybrid retrieval with citation grounding**:
- **Dense retrieval** (semantic search via ChromaDB + sentence embeddings)
- **Sparse retrieval** (keyword search via BM25)
- **Cross-encoder reranking** (BERT-based model scores relevance, returns top 5)
- **Citation grounding** (answers include references to source tickets with metadata)

## Why Hybrid + Reranking + Citations?

- **Dense (embeddings)**: Captures semantic meaning, finds conceptually similar tickets
- **BM25 (sparse)**: Excels at exact keyword matching, finds tickets with specific terms
- **Hybrid**: Combines both for better overall retrieval quality
- **Cross-Encoder Reranking**: Uses BERT-based model to score query-document pairs, fast and accurate
- **Citations**: Provides transparency, allows verification, builds trust in AI answers

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Index data for both methods:
```bash
# Create dense indices (ChromaDB)
python index_data.py

# Create sparse indices (BM25)
python bm25_index.py
```

## Usage

### Command Line

**Dense only (original):**
```bash
python query_data.py "VPN connection issues"
```

**Hybrid retrieval:**
```bash
python query_data.py --hybrid "VPN connection issues"
```

**Hybrid + Cross-encoder reranking (top 5):**
```bash
python query_data.py --hybrid --rerank "VPN connection issues"
```

### API

**Dense only:**
```bash
curl "http://localhost:8000/query?ticket=VPN+connection+issues"
```

**Hybrid retrieval:**
```bash
curl "http://localhost:8000/query?ticket=VPN+connection+issues&use_hybrid=true"
```

**Hybrid + Cross-encoder reranking:**
```bash
curl "http://localhost:8000/query?ticket=VPN+connection+issues&use_hybrid=true&use_reranking=true"
```

**Full pipeline with citations (recommended):**
```bash
curl "http://localhost:8000/query?ticket=VPN+connection+issues&use_hybrid=true&use_reranking=true&use_citations=true"
```

## Example Response with Citations

```json
{
  "ticket": "VPN connection issues",
  "answer": "To fix VPN connection issues, try restarting the VPN client [1] and checking your firewall settings [2]. If the problem persists, verify your network configuration [3].",
  "sources": [
    {
      "citation_id": 1,
      "subject": "VPN Connection Troubleshooting",
      "type": "Incident",
      "priority": "high",
      "queue": "Technical Support"
    },
    {
      "citation_id": 2,
      "subject": "Firewall Configuration Guide",
      "type": "Request",
      "priority": "medium",
      "queue": "IT Support"
    },
    {
      "citation_id": 3,
      "subject": "Network Setup Instructions",
      "type": "Request",
      "priority": "low",
      "queue": "Technical Support"
    }
  ],
  "retrieval_method": "hybrid",
  "reranking_enabled": true,
  "citations_enabled": true
}
```

## How It Works

### Hybrid Retrieval
1. **Query both systems**: Retrieves top-k results from both dense and sparse indices
2. **Normalize scores**: Converts distances/scores to 0-1 range for fair comparison
3. **Combine results**: Merges results using weighted combination (default: 50% dense, 50% sparse)
4. **Rank by combined score**: Returns top results based on hybrid score

### Cross-Encoder Reranking
1. **Takes all hybrid results** (typically 36 documents)
2. **Loads cross-encoder model** (ms-marco-MiniLM-L-6-v2)
3. **Scores each query-document pair** using BERT-based model
4. **Returns top 5** most relevant documents
5. **Uses these for final answer generation**

### Citation Grounding
1. **Numbers each document** [1], [2], [3], etc.
2. **Extracts metadata** (subject, type, priority, queue)
3. **Instructs LLM** to cite sources using [1], [2] format
4. **Returns answer with citations** and source metadata
5. **Users can verify** which tickets informed the answer

## Configuration

In `query_data.py`, you can adjust:
- `dense_weight`: Weight for dense retrieval (default: 0.5)
- `sparse_weight`: Weight for sparse retrieval (default: 0.5)
- `n_per_chunk`: Top-k results per chunk (default: 6)
- `top_k` in reranking: Final documents after reranking (default: 5)
- `use_citations`: Enable/disable citation grounding (default: True)

## Metadata Fields in Citations

Each citation includes:
- `citation_id`: Reference number [1], [2], etc.
- `subject`: Ticket subject/title
- `type`: Ticket type (Incident, Request, etc.)
- `priority`: Priority level (high, medium, low)
- `queue`: Support queue (Technical Support, IT Support, etc.)
- `chunk_id`: Database partition identifier

## Performance Tips

- BM25 indices are stored as pickle files in `data/bm25/`
- Cross-encoder model (~100MB) downloads on first use
- **Dense only**: Fastest, good quality
- **Hybrid**: Moderate speed, better quality
- **Hybrid + Reranking**: Slightly slower, best quality (local inference, no API calls)
- Use reranking when accuracy is critical
- Cross-encoder runs locally (CPU/GPU), no API costs
