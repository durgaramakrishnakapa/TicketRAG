"""
LangGraph agent that:
- Stores API key once and sends it with every ticket query
- Uses the query API to get ticket solutions
- Deduces department (frontend/backend) from the solution and sends email
"""
import os
from typing import Annotated, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, GEMINI_MODEL
from services.query_client import query_ticket_solution
from services.email_sender import send_department_email


# ----- State -----

class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    api_key: Optional[str]


# ----- Tool schemas (for LLM) -----

class SetApiKeyInput(BaseModel):
    api_key: str = Field(description="The Gemini API key for the ticket query API. Store this so the agent can use it for all future ticket queries without asking again.")


class QueryTicketInput(BaseModel):
    ticket: str = Field(description="The ticket string (issue description). Sent to the query API as the first parameter.")


class SendTicketToDepartmentInput(BaseModel):
    department: Literal["frontend", "backend"] = Field(description="The department to send the ticket to: 'frontend' for frontend issues, 'backend' for backend issues.")
    subject: str = Field(description="Short subject line for the email.")
    body: str = Field(description="Email body: include ticket summary and solution from the API.")


# ----- Tool implementations (used by custom tool node with state) -----

def _set_api_key_impl(api_key: str) -> str:
    return "API key stored successfully. I will use it for all ticket queries and will not ask again."


def _query_ticket_impl(api_key: str, ticket: str) -> str:
    if not api_key:
        return "No API key stored. Please provide your API key first so I can query the ticket API."
    return query_ticket_solution(api_key, ticket)


def _send_ticket_to_department_impl(department: str, subject: str, body: str) -> str:
    return send_department_email(department, subject, body)


# ----- Tools exposed to the LLM -----

@tool("set_api_key", args_schema=SetApiKeyInput)
def set_api_key(api_key: str) -> str:
    """Store the user's API key for the ticket query API. Call this when the user provides their API key so you can use it for all future ticket queries without asking again."""
    return _set_api_key_impl(api_key)


@tool("query_ticket", args_schema=QueryTicketInput)
def query_ticket(ticket: str) -> str:
    """Call the query API (POST http://localhost:8000/query) with ticket (1st) and api (2nd) as strings to get the solution for this ticket. Uses the stored API key; call set_api_key first if not set."""
    # Actual call is in call_tool using state["api_key"] + args["ticket"]
    return ""


@tool("send_ticket_to_department", args_schema=SendTicketToDepartmentInput)
def send_ticket_to_department(department: Literal["frontend", "backend"], subject: str, body: str) -> str:
    """Send the ticket and solution to the appropriate department by email. Use 'frontend' for frontend issues (frontend@gmail.com), 'backend' for backend issues (backend@gmail.com). Call this after you have the solution and have deduced the department."""
    return _send_ticket_to_department_impl(department, subject, body)


TOOLS = [set_api_key, query_ticket, send_ticket_to_department]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ----- Custom tool node: inject state (api_key) for query_ticket and set_api_key -----

def call_tool(state: AgentState) -> dict:
    messages = state["messages"]
    api_key = state.get("api_key")
    last_message = messages[-1]

    if not getattr(last_message, "tool_calls", None):
        return {"messages": []}

    outputs = []
    new_api_key = api_key

    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        tool_call_id = tool_call["id"]

        if name == "set_api_key":
            result = _set_api_key_impl(args["api_key"])
            new_api_key = args["api_key"]
            outputs.append(ToolMessage(content=result, tool_call_id=tool_call_id))
        elif name == "query_ticket":
            result = _query_ticket_impl(api_key or "", args["ticket"])
            outputs.append(ToolMessage(content=result, tool_call_id=tool_call_id))
        elif name == "send_ticket_to_department":
            result = _send_ticket_to_department_impl(
                args["department"], args["subject"], args["body"]
            )
            outputs.append(ToolMessage(content=result, tool_call_id=tool_call_id))
        else:
            tool_obj = TOOLS_BY_NAME.get(name)
            if tool_obj:
                result = tool_obj.invoke(args)
                outputs.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
            else:
                outputs.append(ToolMessage(content=f"Unknown tool: {name}", tool_call_id=tool_call_id))

    update: dict = {"messages": outputs}
    if new_api_key is not None:
        update["api_key"] = new_api_key
    return update


# ----- Model and graph -----

SYSTEM_PROMPT = """You are a ticket support agent. You have access to:

1. **set_api_key(api_key)** – When the user first provides their API key for the ticket query API, call this once to store it. After that, never ask for the API key again; use the stored key for every ticket query.

2. **query_ticket(ticket)** – Tool that calls the query API (POST to /query) with two string inputs: ticket (1st) and api (2nd, the stored key). Returns the solution string from past ticket issues and solutions. Use this for every ticket the user submits.

3. **send_ticket_to_department(department, subject, body)** – After you get the solution from the API, deduce whether the issue is **frontend** or **backend** from the solution content, then send an email to the right department:
   - Frontend issues → department "frontend" (frontend@gmail.com)
   - Backend issues → department "backend" (backend@gmail.com)

Workflow for each ticket:
1. If no API key has been stored yet, ask the user for it and call set_api_key. Then proceed.
2. Call query_ticket with the ticket string to get the solution.
3. From the solution text, decide if it is a frontend or backend issue.
4. Call send_ticket_to_department with that department, a short subject, and a body that includes the ticket and solution.
5. Confirm to the user that the email was sent to the appropriate department and summarize what was done.
"""


def create_agent_graph():
    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in the environment for the LangGraph LLM.")

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.2,
        max_retries=4,
        google_api_key=api_key,
    )
    model_with_tools = llm.bind_tools(TOOLS)

    def call_model(state: AgentState):
        response = model_with_tools.invoke(
            [HumanMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        )
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["continue", "end"]:
        messages = state["messages"]
        if not messages:
            return "end"
        last = messages[-1]
        if getattr(last, "tool_calls", None):
            return "continue"
        return "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("llm", call_model)
    workflow.add_node("tools", call_tool)
    workflow.set_entry_point("llm")
    workflow.add_conditional_edges("llm", should_continue, {"continue": "tools", "end": END})
    workflow.add_edge("tools", "llm")

    return workflow.compile()
