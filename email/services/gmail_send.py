"""Gmail API send using OAuth2 credentials (client_secret / credentials.json)."""
import base64
import glob
import os
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import GMAIL_CREDENTIALS_JSON, GMAIL_TOKEN_JSON

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

# Resolve credentials path: env > credentials.json > first client_secret*.json in cwd
def _credentials_path():
    if os.getenv("GMAIL_CREDENTIALS_JSON") and os.path.isfile(os.getenv("GMAIL_CREDENTIALS_JSON", "")):
        return os.getenv("GMAIL_CREDENTIALS_JSON")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, GMAIL_CREDENTIALS_JSON)
    if os.path.isfile(path):
        return path
    for p in glob.glob(os.path.join(root, "client_secret*.json")):
        return p
    return os.path.join(root, GMAIL_CREDENTIALS_JSON)


def _token_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, GMAIL_TOKEN_JSON)


def get_gmail_service():
    """Build Gmail API service; run OAuth flow if no valid token."""
    creds = None
    token_path = _token_path()
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        except Exception:
            pass
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            creds_path = _credentials_path()
            if not os.path.isfile(creds_path):
                return None
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def send_email_via_gmail(to: str, subject: str, body: str, from_email=None) -> str:
    """
    Send one email using Gmail API. from_email ignored (Gmail uses authenticated user).
    Returns status message.
    """
    service = get_gmail_service()
    if not service:
        return "[Gmail] No credentials.json or client_secret*.json found. Add GMAIL_CREDENTIALS_JSON in .env or place credentials in project root."
    msg = MIMEText(body)
    msg["To"] = to
    msg["Subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    try:
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return f"Email sent successfully to {to} (via Gmail API)."
    except HttpError as e:
        return f"Gmail API error sending to {to}: {e}"
