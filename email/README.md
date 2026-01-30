# Ticket Support Agent (LangGraph + Gemini)

An AI agent that:

1. **Stores your API key once** – You provide the Gemini/ticket-query API key at the start; the agent stores it and sends it with every ticket query without asking again.
2. **Uses the query API** – For each ticket (string), the agent calls your `query` endpoint with the stored API key and ticket, and gets back a solution string (from past ticket issues and solutions).
3. **Routes email by department** – From the solution text, the agent deduces whether the issue is **frontend** or **backend**, then sends an email to the right team:
   - **Frontend** → `frontend@gmail.com`
   - **Backend** → `backend@gmail.com`
4. **Notifies you** – It confirms to you which department was emailed and what was done.

Built with **LangGraph** and **Gemini** as the LLM.

## Setup

1. **Environment**

   - `GEMINI_API_KEY` – **Required.** Used by the LangGraph agent as the LLM (Gemini). Get one at [Google AI Studio](https://aistudio.google.com/apikey).
   - `QUERY_API_BASE_URL` – Optional. Base URL of your ticket query API (default: `http://localhost:8000`).
   - For real email sending, set SMTP env vars: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM_EMAIL`. If not set, the agent only logs what it would send (dev mode).

2. **Install**

   ```bash
   pip install -r requirements.txt
   ```

3. **Query API contract**

   Your `query` endpoint should:

   - Accept **POST** with JSON body: `{"api_key": "<key>", "ticket": "<ticket string>"}`.
   - Return a **string** solution, e.g. JSON `{"response": "..."}` or plain string. The agent uses the “solution” text to decide frontend vs backend and to include in the email.

## Run

```bash
export GEMINI_API_KEY="your-gemini-key"
python main.py
```

- First message: provide your **ticket-query API key** (e.g. “My API key is xyz”). The agent stores it and won’t ask again.
- Next messages: send **ticket** strings. The agent will call the query API, infer department, send the email, and tell you the result.
- Say `quit` or `exit` to stop.

## Project layout

- `agent/graph.py` – LangGraph state, tools, and graph (Gemini LLM, custom tool node that uses stored API key).
- `services/query_client.py` – Client for your `query` API.
- `services/email_sender.py` – Sends (or logs) department emails.
- `config.py` – URLs, department emails, SMTP, and `GEMINI_API_KEY`.
- `main.py` – CLI loop to chat with the agent.

## Department emails

Configured in `config.py`:

| Department | Email              |
|-----------|--------------------|
| Frontend  | frontend@gmail.com |
| Backend   | backend@gmail.com  |

Override via code or env if you need different addresses.
