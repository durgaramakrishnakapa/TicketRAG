"""Send emails to department addresses (Gmail API OAuth2 or SMTP or dev log)."""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import (
    DEPARTMENT_EMAILS,
    GMAIL_CREDENTIALS_JSON,
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASSWORD,
    SMTP_FROM_EMAIL,
)


def _has_gmail_credentials():
    """True if Gmail OAuth2 credentials file exists."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.getenv("GMAIL_CREDENTIALS_JSON") and os.path.isfile(os.getenv("GMAIL_CREDENTIALS_JSON", "")):
        return True
    if os.path.isfile(os.path.join(root, GMAIL_CREDENTIALS_JSON)):
        return True
    import glob
    return bool(glob.glob(os.path.join(root, "client_secret*.json")))


def send_department_email(
    department: str,
    subject: str,
    body: str,
) -> str:
    """
    Send an email TO the department address (frontend/backend) FROM your Gmail.
    Recipients are in config.DEPARTMENT_EMAILS; sender is your OAuth-authenticated Gmail.
    Uses Gmail API if credentials exist; else SMTP; else dev log.
    """
    # TO: department address (not your inbox)
    to_address = DEPARTMENT_EMAILS.get(department.lower())
    if not to_address:
        return f"Unknown department: {department}. Use 'frontend' or 'backend'."

    if _has_gmail_credentials():
        try:
            from .gmail_send import send_email_via_gmail
            # FROM: your Gmail (OAuth)  TO: department address
            return send_email_via_gmail(to=to_address, subject=subject, body=body)
        except Exception as e:
            return f"Gmail send failed for {to_address}: {e}"

    if SMTP_HOST and SMTP_USER and SMTP_PASSWORD:
        msg = MIMEMultipart()
        msg["From"] = SMTP_FROM_EMAIL
        msg["To"] = to_address
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_FROM_EMAIL, to_address, msg.as_string())
            return f"Email sent successfully to {to_address} (department: {department})."
        except Exception as e:
            return f"Failed to send email to {to_address}: {e}"

    return (
        f"[DEV] Would send email to {to_address} (department: {department})\n"
        f"Subject: {subject}\n"
        f"Body: {body[:200]}{'...' if len(body) > 200 else ''}"
    )
