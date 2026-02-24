"""
Agentic Orchestrator — LangGraph StateGraph 기반 Plan→Execute→Validate→Replan loop.
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from core.types import AgentState
from config.settings import API_CONFIG, MODEL_CONFIG
from prompts.orchestrator_prompts import (
    PLAN_SYSTEM_PROMPT,
    PLAN_USER_TEMPLATE,
    VALIDATE_SYSTEM_PROMPT,
    VALIDATE_USER_TEMPLATE,
    REPLAN_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


def _create_llm() -> ChatOpenAI:
    """Create the LLM instance for orchestrator."""
    return ChatOpenAI(
        api_key=API_CONFIG.llm_api_key,
        base_url=API_CONFIG.llm_api_url,
        model=MODEL_CONFIG.llm_model,
        temperature=MODEL_CONFIG.temperature,
        max_tokens=MODEL_CONFIG.llm_max_tokens,
    )


def _parse_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response (handles ```json blocks)."""
    import re
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find any JSON-like structure
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


# ── Graph Nodes ───────────────────────────────────────────────────────

def memory_node(state: AgentState) -> dict:
    """Search for similar past messages."""
    from tools.memory_search import search_memory

    t0 = time.time()
    result = search_memory.invoke({"message": state["message"]})
    duration = (time.time() - t0) * 1000

    history = list(state.get("history", []))
    history.append({"step": "memory_search", "duration_ms": round(duration, 1)})

    return {
        "memory_results": result.get("similar_cases", []),
        "history": history,
    }


def plan_node(state: AgentState) -> dict:
    """LLM plans which tools to call."""
    llm = _create_llm()

    memory_str = "없음"
    if state.get("memory_results"):
        cases = state["memory_results"][:3]
        memory_str = "\n".join(
            f"- 메시지: {c['mms'][:100]}... → 정답: {c['correct_entities']}"
            for c in cases
        )

    messages = [
        SystemMessage(content=PLAN_SYSTEM_PROMPT),
        HumanMessage(content=PLAN_USER_TEMPLATE.format(
            message=state["message"],
            memory_results=memory_str,
        )),
    ]

    t0 = time.time()
    response = llm.invoke(messages)
    duration = (time.time() - t0) * 1000

    plan = _parse_json_from_response(response.content)

    history = list(state.get("history", []))
    history.append({"step": "plan", "duration_ms": round(duration, 1), "plan": plan})

    return {"plan": plan.get("plan", []), "history": history}


def execute_node(state: AgentState) -> dict:
    """Execute tools according to the plan."""
    from tools.llm_entity_extractor import llm_extract_entities
    from tools.entity_extractor import extract_entities
    from tools.product_matcher import match_products
    from tools.llm_vocab_filter import llm_filter_matched_products
    from tools.rule_engine_tool import apply_rules
    from tools.schema_transformer import transform_schema

    message = state["message"]
    history = list(state.get("history", []))
    entity_roles = {}
    dag_text = ""

    # Step 1: LLM DAG entity extraction (primary)
    t0 = time.time()
    llm_result = llm_extract_entities.invoke({"message": message})
    candidates = llm_result.get("candidate_entities", [])
    entity_roles = llm_result.get("entity_roles", {})
    dag_text = llm_result.get("dag_text", "")
    history.append({"step": "llm_extract_entities", "duration_ms": round((time.time() - t0) * 1000, 1), "count": len(candidates)})

    # Fallback: Kiwi NLP if LLM returns empty
    if not candidates:
        logger.info("[Execute] LLM returned no entities, falling back to Kiwi NLP")
        t0 = time.time()
        kiwi_result = extract_entities.invoke({"message": message})
        candidates = kiwi_result.get("candidate_entities", [])
        history.append({"step": "extract_entities_fallback", "duration_ms": round((time.time() - t0) * 1000, 1), "count": len(candidates)})

    # Step 2: Apply rules
    t0 = time.time()
    rule_result = apply_rules.invoke({"message": message, "entities": candidates})
    filtered = rule_result.get("filtered_entities", candidates)
    history.append({"step": "apply_rules", "duration_ms": round((time.time() - t0) * 1000, 1), "count": len(filtered)})

    # If replan provided additional entities, merge them
    feedback = state.get("feedback", "")
    if feedback and state.get("validation"):
        missing = state["validation"].get("missing_entities", [])
        if missing:
            for m in missing:
                if m not in filtered:
                    filtered.append(m)
                    logger.info(f"[Execute] added missing entity from feedback: {m}")

    # Step 3: Match products (only offer/unknown entities, skip prerequisite/context/benefit)
    if entity_roles:
        skip_roles = ('prerequisite', 'context', 'benefit')
        matchable = [e for e in filtered if entity_roles.get(e) not in skip_roles]
        skipped = [e for e in filtered if entity_roles.get(e) in skip_roles]
        if skipped:
            logger.info(f"[Execute] Skipping non-offer entities for matching: {skipped}")
    else:
        matchable = filtered

    t0 = time.time()
    match_result = match_products.invoke({"entities": matchable, "message": message})
    matched = match_result.get("matched_products", [])
    history.append({"step": "match_products", "duration_ms": round((time.time() - t0) * 1000, 1), "count": len(matched)})

    # Step 3b: LLM vocabulary filter — remove false positives
    if matched:
        t0 = time.time()
        filter_result = llm_filter_matched_products.invoke({
            "matched_products": matched,
            "message": message,
            "entity_roles": entity_roles,
        })
        matched = filter_result.get("filtered_products", matched)
        removed = filter_result.get("removed_products", [])
        history.append({
            "step": "llm_filter_matched_products",
            "duration_ms": round((time.time() - t0) * 1000, 1),
            "count": len(matched),
            "removed": len(removed),
        })

    # Step 3c: Apply false_positives from previous validation (replan safety net).
    # Prevents the replan loop from repeatedly returning the same over-extracted items
    # that the validator already flagged in a prior iteration.
    if state.get("validation"):
        false_positives = state["validation"].get("false_positives", [])
        if false_positives:
            fp_set = {fp.lower().strip() for fp in false_positives}
            before = len(matched)
            matched = [
                m for m in matched
                if m.get('item_nm', '').lower().strip() not in fp_set
                and m.get('entity', '').lower().strip() not in fp_set
            ]
            if len(matched) < before:
                logger.info(f"[Execute] Removed known false positives: {false_positives}")

    # Step 4: Transform schema
    t0 = time.time()
    schema_result = transform_schema.invoke({"matched_products": matched})
    history.append({"step": "transform_schema", "duration_ms": round((time.time() - t0) * 1000, 1)})

    return {
        "candidate_entities": candidates,
        "filtered_entities": filtered,
        "matched_products": matched,
        "entity_roles": entity_roles,
        "dag_text": dag_text,
        "final_result": schema_result,
        "history": history,
    }


def validate_node(state: AgentState) -> dict:
    """LLM validates the extraction result."""
    llm = _create_llm()

    memory_str = "없음"
    if state.get("memory_results"):
        cases = state["memory_results"][:3]
        memory_str = "\n".join(
            f"- 메시지: {c['mms'][:100]}... → 정답: {c['correct_entities']}"
            for c in cases
        )

    result_str = json.dumps(state.get("final_result", {}), ensure_ascii=False, indent=2)

    messages = [
        SystemMessage(content=VALIDATE_SYSTEM_PROMPT),
        HumanMessage(content=VALIDATE_USER_TEMPLATE.format(
            message=state["message"],
            extraction_result=result_str,
            memory_results=memory_str,
        )),
    ]

    t0 = time.time()
    response = llm.invoke(messages)
    duration = (time.time() - t0) * 1000

    validation = _parse_json_from_response(response.content)

    iteration = state.get("iteration", 0) + 1

    history = list(state.get("history", []))
    history.append({
        "step": "validate",
        "duration_ms": round(duration, 1),
        "is_satisfied": validation.get("is_satisfied", True),
        "score": validation.get("score", 0),
        "iteration": iteration,
    })

    return {
        "validation": validation,
        "iteration": iteration,
        "feedback": validation.get("feedback", ""),
        "history": history,
    }


def should_continue(state: AgentState) -> str:
    """Decide whether to stop or replan."""
    validation = state.get("validation", {})
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if validation.get("is_satisfied", True):
        return "end"
    if iteration >= max_iter:
        logger.info(f"[Orchestrator] max iterations ({max_iter}) reached, stopping")
        return "end"
    return "replan"


def replan_node(state: AgentState) -> dict:
    """Replan based on validation feedback."""
    llm = _create_llm()

    validation = state.get("validation", {})
    result_str = json.dumps(state.get("final_result", {}), ensure_ascii=False, indent=2)

    messages = [
        SystemMessage(content=PLAN_SYSTEM_PROMPT),
        HumanMessage(content=REPLAN_USER_TEMPLATE.format(
            message=state["message"],
            previous_result=result_str,
            feedback=validation.get("feedback", ""),
            missing_entities=validation.get("missing_entities", []),
            false_positives=validation.get("false_positives", []),
        )),
    ]

    t0 = time.time()
    response = llm.invoke(messages)
    duration = (time.time() - t0) * 1000

    plan = _parse_json_from_response(response.content)

    history = list(state.get("history", []))
    history.append({"step": "replan", "duration_ms": round(duration, 1)})

    return {"plan": plan.get("plan", []), "history": history}


# ── Build Graph ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build the orchestrator StateGraph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("memory", memory_node)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("validate", validate_node)
    graph.add_node("replan", replan_node)

    # Add edges
    graph.add_edge(START, "memory")
    graph.add_edge("memory", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "validate")

    # Conditional edge after validate
    graph.add_conditional_edges(
        "validate",
        should_continue,
        {"end": END, "replan": "replan"},
    )
    graph.add_edge("replan", "execute")

    return graph.compile()


# ── Main entry point ──────────────────────────────────────────────────

def process_message(message: str, max_iterations: int = 3) -> dict:
    """
    Process a single MMS message through the agentic pipeline.

    Args:
        message: MMS 광고 메시지 텍스트
        max_iterations: 최대 재시도 횟수

    Returns:
        dict with final_result, history, validation
    """
    graph = build_graph()

    initial_state: AgentState = {
        "message": message,
        "plan": [],
        "memory_results": [],
        "candidate_entities": [],
        "matched_products": [],
        "filtered_entities": [],
        "entity_roles": {},
        "dag_text": "",
        "final_result": None,
        "validation": None,
        "iteration": 0,
        "max_iterations": max_iterations,
        "feedback": "",
        "history": [],
    }

    logger.info(f"[Orchestrator] Processing: {message[:80]}...")
    t0 = time.time()

    final_state = graph.invoke(initial_state)

    total_duration = (time.time() - t0) * 1000
    logger.info(f"[Orchestrator] Done in {total_duration:.0f}ms, {final_state.get('iteration', 0)} iteration(s)")

    return {
        "final_result": final_state.get("final_result", {}),
        "validation": final_state.get("validation", {}),
        "history": final_state.get("history", []),
        "iterations": final_state.get("iteration", 0),
        "total_duration_ms": round(total_duration, 1),
    }
