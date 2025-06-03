"""
Demonstrates LangGraph's fault-tolerance with pending writes.

This example shows how LangGraph handles partial failures in parallel execution:

1. The `finance_assistant` node succeeds and routes to tool execution
2. Two nodes then run in parallel:
   - `data_preprocessor`: Always succeeds and writes results to state
   - `result_analyzer`: Fails on the first attempt only
3. When the super-step fails due to `result_analyzer`, LangGraph:
   - Checkpoints the successful write from `data_preprocessor`
   - On resume, skips the already-completed `data_preprocessor`
   - Re-runs only the failed `result_analyzer` node

This demonstrates LangGraph's ability to preserve partial progress during failures,
avoiding redundant work when resuming from checkpoints.
"""

from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from typing import Annotated, List
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

import json
import os
import sqlite3
import random
import time
import uuid

from dotenv import load_dotenv

load_dotenv()

#  Reproducible randomness
# ------------------------------------------------------------------

random.seed(42)


#  Graph state definition
# ------------------------------------------------------------------


def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dictionaries, with right taking precedence for overlapping keys."""
    if left is None:
        return right
    if right is None:
        return left
    return {**left, **right}


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    attempt_count: int
    finance_data: dict
    intermediate_results: Annotated[dict, merge_dicts]


#  Helper functions & tools
# ------------------------------------------------------------------


def load_finance_data() -> dict:
    """Read finance_data.json located next to this file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "finance_data.json"), "r") as f:
        return json.load(f)


@tool
def get_finance_data():
    """Return the dataset of company contracts (names, amounts, terms, renewals)."""
    return load_finance_data()


@tool
def multiply_by_pi(number: int):
    """Multiply a number by π."""
    return 3.14159 * number


@tool
def simulate_failure_tool():
    """Example tool that *usually* fails (not essential to the demo)."""
    if random.random() < 0.7:
        raise Exception("Simulated tool failure – network timeout!")
    return "Tool executed successfully!"


#  LLM setup and tool binding
# ------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o")
tools = [get_finance_data, multiply_by_pi, simulate_failure_tool]

model_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


#  Helper to decide whether to keep looping after assistant response
# ------------------------------------------------------------------


def should_continue(state: State):
    last_message = state["messages"][-1]
    return "continue" if last_message.tool_calls else "end"


#  Graph nodes
# ------------------------------------------------------------------


def finance_assistant(state: State):
    """LLM‑powered finance expert (no simulated failure here)."""
    attempt_count = state.get("attempt_count", 0) + 1
    state["attempt_count"] = attempt_count

    # Check if parallel processing has been completed
    intermediate_results = state.get("intermediate_results", {})
    has_preprocessed_data = "preprocessed_data" in intermediate_results
    has_analysis_results = "analysis_results" in intermediate_results

    # Extract the user's question from the last human message
    user_question = ""
    for message in reversed(state["messages"]):
        if message.type == "human":
            user_question = message.content
            break

    if has_preprocessed_data and has_analysis_results:
        print(
            f"[INFO] finance_assistant attempt {attempt_count} - providing final response"
        )

        # Final response after parallel processing - use LLM WITHOUT tools to prevent loops
        prompt = f"""You are an expert financial analyst with deep knowledge of business contracts and financial planning.

        The data preprocessing and analysis phases have been completed successfully. You now have access to all the contract data through previous tool calls in this conversation.

        Based on the data you've retrieved, please provide a comprehensive final analysis to answer the user's question: {user_question}

        Do not make any tool calls. Provide your final analysis based on the data already available in the conversation.
        """

        # Use LLM without tools to prevent infinite loops
        response = llm.invoke([SystemMessage(prompt)] + state["messages"])

    else:
        print(f"[INFO] finance_assistant attempt {attempt_count} - making tool calls")

        # Initial response - gather data and prepare for analysis
        prompt = f"""You are an expert financial analyst with deep knowledge of business contracts and financial planning.

        You have access to tools to help analyze financial data:
        - get_finance_data: Retrieves company contract information (amounts, terms, renewals, etc.)
        - multiply_by_pi: Multiplies a number by π when needed for calculations

        The user's question is: {user_question}

        Please use the appropriate tools to gather the data needed and provide a helpful analysis.
        """

        # Use model with tools for initial data gathering
        response = model_with_tools.invoke([SystemMessage(prompt)] + state["messages"])

    return {
        "messages": [response],
        "attempt_count": attempt_count,
        "intermediate_results": {"last_successful_node": "finance_assistant"},
    }


def data_preprocessor(state: State):
    """Always succeeds - its write becomes a *pending write* if peers fail."""
    print("[SUCCESS] data_preprocessor completed")
    preprocessed = {
        "status": "preprocessed",
        "timestamp": time.time(),
    }
    return {
        "intermediate_results": {
            **state.get("intermediate_results", {}),
            "preprocessed_data": preprocessed,
            "last_successful_node": "data_preprocessor",
        }
    }


def result_analyzer(state: State):
    """
    Fails exactly once (the first time this process calls the function) to
    trigger LangGraph's pending‑writes mechanism.
    """
    if not hasattr(result_analyzer, "_has_failed"):
        result_analyzer._has_failed = True
        print("[ERROR] result_analyzer simulated failure")
        raise Exception("Simulated analysis failure (first run)")

    print("[SUCCESS] result_analyzer completed")
    analysis = {
        "status": "analyzed",
        "insights": ["Pattern A detected", "Trend B identified"],
        "timestamp": time.time(),
    }
    return {
        "intermediate_results": {
            **state.get("intermediate_results", {}),
            "analysis_results": analysis,
            "last_successful_node": "result_analyzer",
        }
    }


def convergence_node(state: State):
    """Convergence point after parallel processing completes."""
    print("[INFO] Parallel processing completed, converging results")
    return {
        "intermediate_results": {
            **state.get("intermediate_results", {}),
            "last_successful_node": "convergence_node",
        }
    }


#  Build the graph
# ------------------------------------------------------------------

builder = StateGraph(State)

builder.add_node("finance_assistant", finance_assistant)
builder.add_node("tool_node", tool_node)
builder.add_node("data_preprocessor", data_preprocessor)
builder.add_node("result_analyzer", result_analyzer)
builder.add_node("convergence_node", convergence_node)

builder.add_edge(START, "finance_assistant")
builder.add_conditional_edges(
    "finance_assistant",
    should_continue,
    {"continue": "tool_node", "end": END},
)

builder.add_edge("tool_node", "data_preprocessor")
builder.add_edge("tool_node", "result_analyzer")

# Parallel branches converge to a single node
builder.add_edge("data_preprocessor", "convergence_node")
builder.add_edge("result_analyzer", "convergence_node")

# Then continue to final assistant response
builder.add_edge("convergence_node", "finance_assistant")


#  SQLite checkpointer
# ------------------------------------------------------------------


def setup_checkpointer():
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


graph = builder.compile(checkpointer=setup_checkpointer())


#  Demo runner
# ------------------------------------------------------------------


def demonstrate_fault_tolerance():
    print("\n[DEMO] Fault-tolerance demonstration")
    print("=" * 60)

    # Reset the failure state for the demo
    if hasattr(result_analyzer, "_has_failed"):
        delattr(result_analyzer, "_has_failed")

    # Use a unique thread ID for each run
    thread_id = f"demo-thread-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": thread_id}}

    init_state = {
        "messages": [
            HumanMessage(
                content="Analyze the top 3 contracts by size and multiply the largest by π"
            )
        ],
        "attempt_count": 0,
        "finance_data": {},
        "intermediate_results": {},
    }

    for attempt in range(1, 4):
        print(f"\n[ATTEMPT {attempt}] Starting execution")
        print("-" * 40)
        try:
            if attempt == 1:
                result = graph.invoke(init_state, cfg)
            else:
                print("[RESUME] Resuming from last checkpoint...")
                result = graph.invoke(None, cfg)

            print("\n[SUCCESS] Graph execution completed")
            print("[RESULT] Final state:", result)
            break

        except Exception as err:
            print("[FAILURE] Execution failed:", err)
            state = graph.get_state(cfg)
            print("[CHECKPOINT] Saved at:", state.config)
            print(
                "[STATE] Intermediate results:",
                state.values.get("intermediate_results", {}),
            )
            print("[RETRY] Retrying...")

    print("\n[COMPLETE] Demo finished")
    history = list(graph.get_state_history(cfg))
    for i, cp in enumerate(history[-5:]):
        print(f"  step {cp.metadata.get('step')}: writes → {cp.metadata.get('writes')}")


if __name__ == "__main__":
    demonstrate_fault_tolerance()
