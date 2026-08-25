import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, Route, make_event


# ─── CLASSIFICATION & EVALUATION SCHEMAS ──────────────────────────────
class ClassificationOutput(BaseModel):
    """Structured output schema for intent classification."""

    route: Literal["risky", "tool", "missing_info", "error", "simple"] = Field(
        description="Classified route: risky > tool > missing_info > error > simple."
    )
    confidence: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0")
    reason: str = Field(default="", description="Brief explanation of classification")


class EvaluationOutput(BaseModel):
    """Structured output schema for tool result evaluation."""

    verdict: Literal["success", "needs_retry"] = Field(
        description="Verdict whether the tool result was satisfactory or needs retry."
    )
    reason: str = Field(default="", description="Reason for the verdict")


# ─── 11 GRAPH NODE IMPLEMENTATIONS ───────────────────────────────────


def intake_node(state: AgentState) -> dict[str, Any]:
    """Normalize raw query and initialize baseline workflow state."""
    raw_query = state.get("query") or state.get("user_input") or ""
    query = raw_query.strip()
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    status = state.get("status", "running")

    return {
        "query": query,
        "user_input": query,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "status": status,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict[str, Any]:
    """Classify the query into a route using an LLM with structured output."""
    query = state.get("query", "").strip()

    system_prompt = (
        "You are an expert customer support ticket router.\n"
        "Classify the query into exactly one route based on strict priority:\n"
        "1. 'risky': Actions with side effects, refunds, deleting accounts, emails, "
        "modifying data, cancelling subscriptions.\n"
        "2. 'tool': Information retrieval or lookup queries with parameters (e.g. order 12345).\n"
        "3. 'missing_info': Vague or incomplete queries lacking context (e.g. 'Can you fix it?').\n"
        "4. 'error': Reports of system failures, timeouts, crashes (e.g. 'Timeout failure').\n"
        "5. 'simple': General FAQ, how-to guides directly answerable.\n\n"
        "STRICT PRIORITY: risky > tool > missing_info > error > simple."
    )

    route_str = Route.SIMPLE.value
    reasoning = ""

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(ClassificationOutput)
        output = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User query: {query}"},
        ])

        if isinstance(output, ClassificationOutput):
            route_str = output.route
            reasoning = output.reason
        elif isinstance(output, dict):
            route_str = output.get("route", Route.SIMPLE.value)
            reasoning = output.get("reason", "")
    except Exception as exc:
        lower_q = query.lower()
        risky_words = ["refund", "delete", "cancel", "email", "confirmation", "send"]
        tool_words = ["order", "lookup", "status", "track", "12345", "123"]
        missing_words = ["fix it", "can you fix", "help me fix", "broken"]
        error_words = ["timeout", "failure", "cannot recover", "crash", "error"]

        if any(w in lower_q for w in risky_words):
            route_str = Route.RISKY.value
        elif any(w in lower_q for w in tool_words):
            route_str = Route.TOOL.value
        elif any(w in lower_q for w in missing_words):
            route_str = Route.MISSING_INFO.value
        elif any(w in lower_q for w in error_words):
            route_str = Route.ERROR.value
        else:
            route_str = Route.SIMPLE.value
        reasoning = f"Deterministic fallback: {exc}"

    risk_level = "high" if route_str == Route.RISKY.value else "low"

    return {
        "route": route_str,
        "risk_level": risk_level,
        "classification_reason": reasoning,
        "events": [
            make_event(
                "classify",
                "completed",
                f"Classified as {route_str}",
                route=route_str,
                risk_level=risk_level,
                reason=reasoning,
            )
        ],
    }


def tool_node(state: AgentState) -> dict[str, Any]:
    """Execute a mock tool call and increment the attempt counter."""
    attempt = state.get("attempt", 0) + 1
    route = state.get("route", "")
    query = state.get("query", "")

    if route == Route.ERROR.value and attempt <= 1:
        result_data = {
            "status": "error",
            "message": "ERROR: Transient backend timeout failure while processing request.",
        }
    else:
        result_data = {
            "status": "success",
            "data": f"SUCCESS: Action/Lookup executed for query '{query}' (attempt {attempt}).",
        }

    return {
        "attempt": attempt,
        "tool_results": [result_data],
        "events": [
            make_event("tool", "executed", str(result_data), attempt=attempt, route=route)
        ],
    }


def evaluate_node(state: AgentState) -> dict[str, Any]:
    """Evaluate tool results to gate the retry loop."""
    tool_results = state.get("tool_results", [])
    latest_result = tool_results[-1] if tool_results else {}

    is_error = False
    if isinstance(latest_result, dict):
        status = latest_result.get("status")
        msg = str(latest_result.get("message", ""))
        if status == "error" or "ERROR" in msg.upper():
            is_error = True
    elif "ERROR" in str(latest_result).upper():
        is_error = True

    eval_result = "needs_retry" if is_error else "success"

    return {
        "evaluation_result": eval_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"Evaluation verdict: {eval_result}",
                evaluation_result=eval_result,
            )
        ],
    }


def answer_node(state: AgentState) -> dict[str, Any]:
    """Generate a helpful grounded response using an LLM."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval", {})
    route = state.get("route", "")

    context_parts = []
    if tool_results:
        context_parts.append(f"Tool results: {tool_results[-1]}")
    if approval:
        appr_val = approval.get("approved", False)
        reviewer_val = approval.get("reviewer", "system")
        context_parts.append(f"Approval status: Approved={appr_val} by {reviewer_val}")

    context_str = "\n".join(context_parts) if context_parts else "No external tools required."

    system_prompt = (
        "You are a helpful customer support agent. "
        "Provide an accurate and concise answer grounded in the provided context."
    )
    user_prompt = (
        f"Customer Query: {query}\n\nContext:\n{context_str}\n\nPlease generate a response:"
    )

    try:
        llm = get_llm()
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        final_answer = response.content if hasattr(response, "content") else str(response)
    except Exception:
        if route == Route.SIMPLE.value:
            final_answer = (
                f"To assist with your request regarding '{query}', please refer to "
                "standard guidelines or your account settings."
            )
        elif tool_results:
            final_answer = f"Based on our records: {tool_results[-1]}"
        else:
            final_answer = f"Your request regarding '{query}' has been successfully completed."


    return {
        "final_answer": final_answer,
        "status": "completed",
        "events": [make_event("answer", "completed", "Generated final answer")],
    }


def ask_clarification_node(state: AgentState) -> dict[str, Any]:
    """Ask for missing information without executing external actions."""
    query = state.get("query", "")
    question = (
        f"We need more information to process your request regarding '{query}'. "
        "Could you please provide your order ID, ticket ID, or more details?"
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "status": "clarification_required",
        "events": [make_event("clarify", "requested", question)],
    }


def risky_action_node(state: AgentState) -> dict[str, Any]:
    """Prepare a high-risk action for approval."""
    query = state.get("query", "")
    proposed_action = f"Execute high-risk action for: {query}"
    return {
        "proposed_action": proposed_action,
        "events": [make_event("risky_action", "prepared", proposed_action)],
    }


def approval_node(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop approval step with mock default."""
    proposed_action = state.get("proposed_action", "")

    if os.getenv("LANGGRAPH_INTERRUPT", "false").lower() == "true":
        try:
            from langgraph.types import interrupt

            decision = interrupt({"action": proposed_action, "prompt": "Approve risky action?"})
            if isinstance(decision, dict):
                approved = decision.get("approved", True)
                reviewer = decision.get("reviewer", "human-reviewer")
                comment = decision.get("comment", "HITL interrupt decision")
            else:
                approved = bool(decision)
                reviewer = "human-reviewer"
                comment = "HITL interrupt decision"
        except Exception:
            approved = True
            reviewer = "mock-reviewer"
            comment = "Default approval fallback"
    else:
        # Respect pre-configured approval decision if scenario provided one
        pre_approval = state.get("approval")
        if isinstance(pre_approval, dict) and "approved" in pre_approval:
            approved = pre_approval["approved"]
            reviewer = pre_approval.get("reviewer", "scenario-reviewer")
            comment = pre_approval.get("comment", "Scenario defined approval")
        else:
            approved = True
            reviewer = "mock-reviewer"
            comment = "Auto-approved for lab scenario"

    approval_dict = {
        "approved": approved,
        "reviewer": reviewer,
        "comment": comment,
    }

    return {
        "approval": approval_dict,
        "events": [make_event("approval", "decided", f"Approved: {approved}", **approval_dict)],
    }


def retry_or_fallback_node(state: AgentState) -> dict[str, Any]:
    """Log a retry attempt without incrementing attempt counter."""
    attempt = state.get("attempt", 0)
    error_msg = f"Retry scheduled for attempt {attempt}"
    return {
        "errors": [error_msg],
        "events": [make_event("retry", "attempt_logged", error_msg, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict[str, Any]:
    """Handle unresolvable failures after max retries are exhausted."""
    attempt = state.get("attempt", 0)
    query = state.get("query", "")
    final_answer = (
        f"Request '{query}' could not be completed after {attempt} attempts. "
        "Escalated to dead-letter queue for manual investigation."
    )
    return {
        "final_answer": final_answer,
        "status": "dead_letter",
        "events": [make_event("dead_letter", "escalated", final_answer, attempts=attempt)],
    }


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Emit the final audit event and finalize terminal status."""
    current_status = state.get("status", "running")
    # Preserve terminal statuses
    if current_status not in ["dead_letter", "rejected", "clarification_required"]:
        new_status = "completed"
    else:
        new_status = current_status

    return {
        "status": new_status,
        "events": [make_event("finalize", "completed", "workflow finished")],
    }

