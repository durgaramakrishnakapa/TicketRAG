"""Client for the ticket query API."""
import requests

from config import QUERY_API_BASE_URL, QUERY_API_PATH


def query_ticket_solution(api_key: str, ticket: str) -> str:
    """
    Call the query API: GET {base}/query?ticket=...&api_key=...
    Returns solution string.
    """
    url = f"{QUERY_API_BASE_URL.rstrip('/')}{QUERY_API_PATH}"
    try:
        resp = requests.get(
            url,
            params={"ticket": ticket, "api_key": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            return resp.text
        if isinstance(data, str):
            return data
        return data.get("response", data.get("solution", str(data)))
    except requests.RequestException as e:
        return f"Query API error: {e}"
    except Exception as e:
        return f"Error: {e}"
