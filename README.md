# TicketRAG - AI-Powered Support Ticket System

An intelligent support ticket system that uses advanced RAG (Retrieval-Augmented Generation) to automatically resolve customer support tickets and route them to the appropriate department.

## Overview

TicketRAG analyzes support tickets, finds similar past tickets from a knowledge base of 100,000+ historical tickets, generates AI-powered solutions using Google Gemini, and automatically emails the appropriate department with ticket details and suggested resolutions.

## Key Features

- **Hybrid Retrieval**: Combines dense (semantic) and sparse (keyword) search for optimal results
- **Cross-Encoder Reranking**: BERT-based model scores relevance for top 5 most relevant documents
- **Citation Grounding**: Answers include references to source tickets with metadata
- **Intelligent Routing**: AI classifies tickets as frontend/backend and emails appropriate team
- **Scalable**: Parallel querying across 6 database partitions

## Architecture

**Ticket API (RAG Backend)**
- FastAPI server with ChromaDB vector database
- Hybrid retrieval: Dense (sentence embeddings) + Sparse (BM25)
- Cross-encoder reranking for precision
- Google Gemini for answer generation

**Email Agent (LangGraph Orchestrator)**
- CLI interface for ticket submission
- LangGraph agent with conversation state
- Automatic department classification
- Gmail/SMTP email delivery

## Quick Start

### 1. Install Dependencies

```bash
# Ticket API
cd ticket_api
pip install -r requirements.txt

# Email Agent
cd ../email
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` files in both directories:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Index Data

```bash
cd ticket_api
python index_data.py      # Dense indices
python bm25_index.py      # Sparse indices
```

### 4. Run

**Start API:**
```bash
cd ticket_api
python api.py
```

**Start Agent:**
```bash
cd email
python main.py
```

## Usage

### API Query

```bash
curl "http://localhost:8000/query?ticket=VPN+issues&use_hybrid=true&use_reranking=true&use_citations=true"
```

### Agent CLI

```
You: My API key is abc123xyz
Agent: API key stored successfully.

You: The login button is not responding
Agent: This is a frontend issue. Email sent to frontend team with solution from similar past tickets.
```

## Configuration

**Department Emails** (`email/config.py`):
```python
DEPARTMENT_EMAILS = {
    "frontend": "frontend-team@example.com",
    "backend": "backend-team@example.com",
}
```

**Retrieval Parameters** (`ticket_api/query_data.py`):
```python
TOP_K_PER_CHUNK = 6      # Documents per partition
dense_weight = 0.5       # Dense retrieval weight
sparse_weight = 0.5      # Sparse retrieval weight
```

## How It Works

1. **Hybrid Retrieval**: Query searches both semantic (ChromaDB) and keyword (BM25) indices
2. **Score Fusion**: Combines and normalizes scores from both methods (36 documents)
3. **Reranking**: Cross-encoder scores all results, returns top 5
4. **Answer Generation**: Gemini generates response with citations [1], [2], etc.
5. **Classification**: Agent determines frontend vs backend from solution
6. **Email Routing**: Sends ticket + solution to appropriate department

## Project Structure

```
ticket_flow/
├── ticket_api/          # RAG Backend
│   ├── api.py          # FastAPI server
│   ├── query_data.py   # Hybrid retrieval logic
│   ├── index_data.py   # Dense indexing
│   ├── bm25_index.py   # Sparse indexing
│   └── data/           # Datasets and indices
│
└── email/              # Email Agent
    ├── main.py         # CLI interface
    ├── config.py       # Configuration
    ├── agent/          # LangGraph agent
    └── services/       # Email & API clients
```

## Tech Stack

- **Backend**: FastAPI, ChromaDB, BM25, Sentence Transformers
- **Agent**: LangGraph, LangChain, Google Gemini
- **Email**: Gmail API (OAuth2) / SMTP

## Performance

- Retrieval: ~2-3 seconds (hybrid search, 100K+ tickets)
- Reranking: ~1-2 seconds (cross-encoder)
- Total: ~5-7 seconds end-to-end

---


