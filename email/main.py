"""
Run the ticket support agent. Provide your API key once; then submit tickets.
The agent will query the API for solutions, deduce frontend/backend, and send the email.
"""
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env from project folder so GEMINI_API_KEY and other vars are set
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from langchain_core.messages import HumanMessage

from agent import create_agent_graph

try:
    from langchain_google_genai import ChatGoogleGenerativeAIError
except ImportError:
    ChatGoogleGenerativeAIError = Exception


def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY in the environment (used for the LangGraph LLM).")
        sys.exit(1)

    graph = create_agent_graph()
    # Full conversation state: messages list and stored api_key
    state: dict = {"messages": [], "api_key": None}

    print(
        "Ticket support agent ready.\n"
        "  - First time: send your API key for the ticket query API (e.g. 'My API key is xxx').\n"
        "  - Then send a ticket string to get a solution and auto-send email to the right department.\n"
        "  - Say 'quit' or 'exit' to stop.\n"
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        # Append user message and run graph with full state
        state["messages"] = list(state["messages"]) + [HumanMessage(content=user_input)]

        try:
            for chunk in graph.stream(state, stream_mode="values"):
                state = {**state, **chunk}
        except ChatGoogleGenerativeAIError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                # Remove the message we just added so user can retry
                state["messages"] = state["messages"][:-1]
                print(
                    "Agent: Gemini API quota/rate limit reached. "
                    "Try again in a minute. Using gemini-2.5-flash by default; set GEMINI_MODEL in .env to override."
                )
            else:
                raise

        # Show last assistant text response (handle Gemini list-of-parts format)
        messages = state["messages"]
        for m in reversed(messages):
            if getattr(m, "type", "") == "ai" and getattr(m, "content", None):
                content = m.content
                if isinstance(content, list):
                    parts = [p.get("text", str(p)) for p in content if isinstance(p, dict) and p.get("text")]
                    text = " ".join(parts) if parts else str(content)
                else:
                    text = content
                print(f"Agent: {text}")
                break
        else:
            print("Agent: [Done]")

    print("Goodbye.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "run":
        print("Usage: python main.py [run]")
        sys.exit(0)
    main()
