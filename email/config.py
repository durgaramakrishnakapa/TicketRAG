"""Configuration for the ticket agent."""
import os

# Query API (your existing endpoint)
QUERY_API_BASE_URL = os.getenv("QUERY_API_BASE_URL", "http://localhost:8000")
QUERY_API_PATH = "/query"

# Department recipients: mail is sent TO these addresses FROM your Gmail (OAuth account)
DEPARTMENT_EMAILS = {
    "frontend": "n210888@rguktn.ac.in",
    "backend": "n210972@rguktn.ac.in",
}

# Optional: SMTP for sending emails (if not set, emails are logged only)
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "agent@example.com")

# Gmail API (OAuth2): path to client secret JSON; token.json is created after first auth
GMAIL_CREDENTIALS_JSON = os.getenv("GMAIL_CREDENTIALS_JSON", "credentials.json")
GMAIL_TOKEN_JSON = os.getenv("GMAIL_TOKEN_JSON", "token.json")

# Gemini API key for the LangGraph LLM (separate from the user-provided key for the query API)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Model name: use GEMINI_MODEL in .env to override
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
