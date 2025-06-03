"""
Demonstrates LangGraph's retry logic and fallback strategies.

This example shows how to handle unreliable nodes with intelligent retry patterns:

1. A node attempts to process data but fails frequently (simulated)
2. On failure, retry the same operation up to N times
3. After N failures, try a fallback strategy:
   - Option A: Simplify the input and retry
   - Option B: Skip the problematic processing entirely
4. Continue with the workflow using whatever data is available

This demonstrates resilient workflow design for unreliable external services,
API timeouts, or any operation that might fail intermittently.
"""

from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from typing import Annotated, Optional
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

import random
import time
import uuid
import sqlite3

from dotenv import load_dotenv

load_dotenv()

# Reproducible randomness

random.seed(42)


#  Graph state definition
# ------------------------------------------------------------------


class RetryState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    input_data: str
    processed_data: Optional[str]
    retry_count: int
    max_retries: int
    fallback_used: bool
    processing_complete: bool


#  Graph nodes
# ------------------------------------------------------------------


def data_preparation(state: RetryState):
    """Prepares input data for processing."""
    print("[PREP] Preparing data for processing")

    # Extract user request from messages
    user_request = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            user_request = msg.content
            break

    # Simulate preparing complex data based on user request
    input_data = f"complex_analysis_request:{user_request}"

    return {
        "input_data": input_data,
        "retry_count": 0,
        "max_retries": 3,
        "fallback_used": False,
        "processing_complete": False,
    }


def unreliable_processor(state: RetryState):
    """
    Simulates an unreliable external service or API call.
    Fails 95% of the time to demonstrate retry logic and fallbacks.
    """
    retry_count = state.get("retry_count", 0)
    input_data = state.get("input_data", "")

    print(f"[PROCESS] Attempting to process data (attempt {retry_count + 1})")
    print(f"[INPUT] {input_data}")

    # Simulate processing failure (95% failure rate for demo purposes)
    if random.random() < 0.95:
        print("[FAILURE] Processing failed - simulated service timeout")
        # Instead of raising exception, return state indicating failure
        return {
            "processing_complete": False,  # This will trigger retry logic
        }

    # Success case
    print("[SUCCESS] Processing completed successfully")
    processed_result = f"PROCESSED_SUCCESSFULLY:{input_data}:timestamp_{time.time()}"

    return {
        "processed_data": processed_result,
        "processing_complete": True,
    }


def retry_logic(state: RetryState):
    """
    Handles retry logic and fallback strategies.

    Strategy:
    1. Retry same operation up to max_retries times
    2. If still failing, try simplified input (fallback strategy A)
    3. If that fails, skip processing entirely (fallback strategy B)
    """
    retry_count = state.get("retry_count", 0) + 1
    max_retries = state.get("max_retries", 3)
    input_data = state.get("input_data", "")

    print(f"[RETRY] Handling failure (retry count: {retry_count})")

    if retry_count <= max_retries:
        # Strategy 1: Retry same operation
        print(f"[RETRY] Attempting retry {retry_count}/{max_retries}")
        return {
            "retry_count": retry_count,
        }

    elif retry_count <= max_retries + 2:
        # Strategy 2: Fallback A - Simplify input and retry
        print("[FALLBACK A] Simplifying input and retrying")
        simplified_input = input_data.replace("complex_", "simple_")

        return {
            "input_data": simplified_input,
            "retry_count": retry_count,
            "fallback_used": True,
        }

    else:
        # Strategy 3: Fallback B - Skip processing entirely
        print("[FALLBACK B] Giving up on processing, using default result")
        default_result = (
            f"PROCESSING_SKIPPED:{input_data}:used_fallback_at_{time.time()}"
        )

        return {
            "processed_data": default_result,
            "processing_complete": True,
            "fallback_used": True,
        }


def response_generator(state: RetryState):
    """Generates the final user-facing response based on processing outcome."""
    processed_data = state.get("processed_data", "")
    fallback_used = state.get("fallback_used", False)

    if fallback_used:
        print("[COMPLETE] Workflow completed using fallback strategy")
        status = "completed with fallback"
    else:
        print("[COMPLETE] Workflow completed successfully")
        status = "completed successfully"

    # Generate appropriate response based on outcome
    if "PROCESSED_SUCCESSFULLY" in processed_data:
        response_content = "✅ Data processing completed successfully! Your request has been fully processed."
    elif "PROCESSING_SKIPPED" in processed_data:
        response_content = "⚠️ Data processing was skipped due to service issues, but we've provided a default response."
    else:
        response_content = "🔄 Data processing completed with alternative method."

    response = AIMessage(content=response_content)

    return {
        "messages": [response],
    }


#  Routing functions
# ------------------------------------------------------------------


def route_after_processing(state: RetryState):
    """Routes based on whether processing completed successfully."""
    if state.get("processing_complete", False):
        return "response_generator"
    else:
        return "retry_logic"


def route_after_retry(state: RetryState):
    """Routes based on retry logic decision."""
    if state.get("processing_complete", False):
        return "response_generator"
    else:
        return "unreliable_processor"


#  Build the graph
# ------------------------------------------------------------------

builder = StateGraph(RetryState)

builder.add_node("data_preparation", data_preparation)
builder.add_node("unreliable_processor", unreliable_processor)
builder.add_node("retry_logic", retry_logic)
builder.add_node("response_generator", response_generator)

# Linear flow with retry loops
builder.add_edge(START, "data_preparation")
builder.add_edge("data_preparation", "unreliable_processor")

# Conditional routing based on processing outcome
builder.add_conditional_edges(
    "unreliable_processor",
    route_after_processing,
    {
        "retry_logic": "retry_logic",
        "response_generator": "response_generator",
    },
)

# Retry loop or completion
builder.add_conditional_edges(
    "retry_logic",
    route_after_retry,
    {
        "unreliable_processor": "unreliable_processor",
        "response_generator": "response_generator",
    },
)

builder.add_edge("response_generator", END)


#  SQLite checkpointer
# ------------------------------------------------------------------


def setup_checkpointer():
    conn = sqlite3.connect("retry_checkpoints.db", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


graph = builder.compile(checkpointer=setup_checkpointer())


#  Demo runner
# ------------------------------------------------------------------


def demonstrate_retry_logic():
    print("\n[DEMO] Retry Logic and Fallback Strategies")
    print("=" * 60)

    # Use a unique thread ID for each demo run
    thread_id = f"retry-demo-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": thread_id}}

    init_state = {
        "messages": [
            HumanMessage(
                content="Please analyze this complex dataset and provide insights"
            )
        ],
        "input_data": "",
        "processed_data": None,
        "retry_count": 0,
        "max_retries": 3,
        "fallback_used": False,
        "processing_complete": False,
    }

    try:
        print("\n[START] Beginning workflow with unreliable processing")
        print("-" * 50)

        result = graph.invoke(init_state, cfg)

        print("\n[RESULT] Final workflow state:")
        print(f"  Processing Complete: {result.get('processing_complete', False)}")
        print(f"  Fallback Used: {result.get('fallback_used', False)}")
        print(f"  Total Retry Count: {result.get('retry_count', 0)}")
        print(f"  Final Data: {result.get('processed_data', 'None')[:100]}...")

        # Show final message
        if result.get("messages"):
            final_message = result["messages"][-1]
            print(f"  Final Response: {final_message.content}")

    except Exception as err:
        print(f"[ERROR] Workflow failed completely: {err}")

    print("\n[COMPLETE] Retry demonstration finished")


if __name__ == "__main__":
    demonstrate_retry_logic()
