"""
Product Trace Report - Report Generation Utilities
====================================================

Utilities for generating human-readable and machine-readable trace reports
for product extraction analysis.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class StepTrace:
    """
    Trace data for a single workflow step.

    Attributes:
        step_name: Name of the step (e.g., 'InputValidationStep')
        step_number: Sequential step number (1-9)
        duration_seconds: Time taken to execute this step
        status: 'success' or 'failed'
        input_data: Relevant input data for this step
        output_data: Relevant output data from this step
        product_changes: Product-specific details and transformations
    """
    step_name: str
    step_number: int
    duration_seconds: float
    status: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    product_changes: Dict[str, Any] = field(default_factory=dict)
    substep_timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class TraceResult:
    """
    Complete trace result for a product extraction run.

    Attributes:
        message: Original MMS message
        message_id: Message identifier
        timestamp: When the trace was captured
        total_duration: Total processing time in seconds
        step_traces: List of per-step trace data
        final_products: Final extracted product list
        similarity_scores: DataFrame with similarity calculations (Step 7)
        llm_prompt: The prompt sent to LLM (Step 5)
        llm_response: Raw LLM response (Step 5)
        extractor_config: Configuration used for extraction
    """
    message: str
    message_id: str
    timestamp: datetime
    total_duration: float
    step_traces: List[StepTrace] = field(default_factory=list)
    final_products: List[Dict] = field(default_factory=list)
    similarity_scores: Optional[pd.DataFrame] = None
    llm_prompt: str = ""
    llm_response: str = ""
    extractor_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace result to dictionary for JSON serialization."""
        result = {
            'message': self.message,
            'message_id': self.message_id,
            'timestamp': self.timestamp.isoformat(),
            'total_duration': self.total_duration,
            'step_traces': [asdict(st) for st in self.step_traces],
            'final_products': self.final_products,
            'llm_prompt': self.llm_prompt,
            'llm_response': self.llm_response,
            'extractor_config': self.extractor_config
        }

        # Convert similarity_scores DataFrame to dict if present
        if self.similarity_scores is not None and not self.similarity_scores.empty:
            result['similarity_scores'] = self.similarity_scores.to_dict(orient='records')
        else:
            result['similarity_scores'] = []

        return result


def _truncate(text: str, max_len: int = 500) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _format_list(items: List, max_items: int = 10) -> str:
    """Format a list for display, truncating if needed."""
    if not items:
        return "[]"
    if len(items) <= max_items:
        return str(items)
    return f"{items[:max_items]}... ({len(items)} total)"


def _format_dict(d: Dict, indent: int = 2) -> str:
    """Format a dictionary for display with indentation."""
    try:
        return json.dumps(d, ensure_ascii=False, indent=indent)
    except (TypeError, ValueError):
        return str(d)


def _format_dataframe_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Format DataFrame as ASCII table."""
    if df is None or df.empty:
        return "  (No data)"

    display_df = df.head(max_rows)

    # Build ASCII table
    lines = []

    # Get column widths
    col_widths = {}
    for col in display_df.columns:
        max_width = max(
            len(str(col)),
            display_df[col].astype(str).str.len().max() if len(display_df) > 0 else 0
        )
        col_widths[col] = min(max_width, 20)  # Cap at 20 chars

    # Header
    header = "  | " + " | ".join(
        str(col)[:col_widths[col]].ljust(col_widths[col])
        for col in display_df.columns
    ) + " |"

    separator = "  +" + "+".join(
        "-" * (col_widths[col] + 2)
        for col in display_df.columns
    ) + "+"

    lines.append(separator)
    lines.append(header)
    lines.append(separator)

    # Data rows
    for _, row in display_df.iterrows():
        row_str = "  | " + " | ".join(
            str(row[col])[:col_widths[col]].ljust(col_widths[col])
            for col in display_df.columns
        ) + " |"
        lines.append(row_str)

    lines.append(separator)

    if len(df) > max_rows:
        lines.append(f"  ... ({len(df) - max_rows} more rows)")

    return "\n".join(lines)


def generate_text_report(trace: TraceResult) -> str:
    """
    Generate a human-readable text report from trace result.

    Args:
        trace: TraceResult with captured data

    Returns:
        Formatted text report string
    """
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("PRODUCT EXTRACTION TRACE REPORT")
    lines.append("=" * 80)
    lines.append(f"Message ID: {trace.message_id}")
    lines.append(f"Message: \"{_truncate(trace.message, 100)}\" ({len(trace.message)} chars)")
    lines.append(f"Timestamp: {trace.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Duration: {trace.total_duration:.2f} seconds")
    lines.append("")

    # Extractor Configuration
    if trace.extractor_config:
        lines.append("-" * 80)
        lines.append("EXTRACTOR CONFIGURATION")
        lines.append("-" * 80)
        for key, value in trace.extractor_config.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

    # Step-by-step traces
    for step_trace in trace.step_traces:
        status_icon = {"success": "OK", "skipped": "SKIP", "failed": "FAIL"}.get(step_trace.status, "FAIL")
        lines.append("-" * 80)
        lines.append(f"STEP {step_trace.step_number}: {step_trace.step_name} "
                    f"({step_trace.duration_seconds:.2f}s) [{status_icon}]")
        lines.append("-" * 80)

        # Input section
        if step_trace.input_data:
            lines.append("INPUT:")
            for key, value in step_trace.input_data.items():
                if isinstance(value, pd.DataFrame):
                    lines.append(f"  {key} (DataFrame): shape={value.shape}")
                elif isinstance(value, (list, dict)):
                    lines.append(f"  {key}: {_truncate(_format_dict(value) if isinstance(value, dict) else str(value), 200)}")
                else:
                    lines.append(f"  {key}: {_truncate(str(value), 200)}")
            lines.append("")

        # Output section
        if step_trace.output_data:
            lines.append("OUTPUT:")
            for key, value in step_trace.output_data.items():
                if isinstance(value, pd.DataFrame):
                    lines.append(f"  {key} (DataFrame): shape={value.shape}")
                    if not value.empty and key in ['similarities_fuzzy', 'cand_entities_sim']:
                        lines.append(_format_dataframe_table(value.head(5)))
                elif isinstance(value, (list, dict)):
                    lines.append(f"  {key}: {_truncate(_format_dict(value) if isinstance(value, dict) else str(value), 300)}")
                else:
                    lines.append(f"  {key}: {_truncate(str(value), 300)}")
            lines.append("")

        # Product focus section
        if step_trace.product_changes:
            lines.append("PRODUCT FOCUS:")
            for key, value in step_trace.product_changes.items():
                if isinstance(value, pd.DataFrame):
                    lines.append(f"  {key}:")
                    lines.append(_format_dataframe_table(value))
                elif isinstance(value, (list, dict)):
                    lines.append(f"  {key}: {_format_dict(value) if isinstance(value, dict) else str(value)}")
                else:
                    lines.append(f"  {key}: {value}")
            lines.append("")

    # Similarity Analysis (Step 7 detail)
    if trace.similarity_scores is not None and not trace.similarity_scores.empty:
        lines.append("-" * 80)
        lines.append("SIMILARITY ANALYSIS (Step 7 Detail)")
        lines.append("-" * 80)
        lines.append(_format_dataframe_table(trace.similarity_scores))
        lines.append("")

    # LLM Prompt and Response (Step 5 detail)
    if trace.llm_prompt:
        lines.append("-" * 80)
        lines.append("LLM PROMPT (Step 5)")
        lines.append("-" * 80)
        lines.append(_truncate(trace.llm_prompt, 2000))
        lines.append("")

    if trace.llm_response:
        lines.append("-" * 80)
        lines.append("LLM RESPONSE (Step 5)")
        lines.append("-" * 80)
        lines.append(_truncate(trace.llm_response, 2000))
        lines.append("")

    # Final Products Summary
    lines.append("=" * 80)
    lines.append(f"FINAL PRODUCTS: {len(trace.final_products)}")
    lines.append("=" * 80)

    for i, product in enumerate(trace.final_products, 1):
        item_nm = product.get('item_nm', 'Unknown')
        item_ids = product.get('item_id', ['#'])
        item_names_in_msg = product.get('item_name_in_msg', [])
        expected_actions = product.get('expected_action', ['Unknown'])

        item_id_str = ', '.join(str(x) for x in item_ids) if isinstance(item_ids, list) else str(item_ids)
        action_str = ', '.join(str(x) for x in expected_actions) if isinstance(expected_actions, list) else str(expected_actions)

        lines.append(f"{i}. {item_nm} ({item_id_str})")
        lines.append(f"   - Names in message: {item_names_in_msg}")
        lines.append(f"   - Expected action: {action_str}")

    if not trace.final_products:
        lines.append("  (No products extracted)")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_json_report(trace: TraceResult, indent: int = 2) -> str:
    """
    Generate a JSON report from trace result.

    Args:
        trace: TraceResult with captured data
        indent: JSON indentation level

    Returns:
        JSON string
    """
    return json.dumps(trace.to_dict(), ensure_ascii=False, indent=indent)


def generate_markdown_report(trace: TraceResult) -> str:
    """
    Generate a Markdown report from trace result.

    Args:
        trace: TraceResult with captured data

    Returns:
        Markdown formatted string
    """
    lines = []

    # Header
    lines.append("# Product Extraction Trace Report")
    lines.append("")
    lines.append(f"**Message ID:** `{trace.message_id}`")
    lines.append(f"**Timestamp:** {trace.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Duration:** {trace.total_duration:.2f} seconds")
    lines.append("")
    lines.append("## Message")
    lines.append(f"```")
    lines.append(trace.message)
    lines.append(f"```")
    lines.append("")

    # Configuration
    if trace.extractor_config:
        lines.append("## Extractor Configuration")
        lines.append("| Setting | Value |")
        lines.append("|---------|-------|")
        for key, value in trace.extractor_config.items():
            lines.append(f"| {key} | `{value}` |")
        lines.append("")

    # Step traces
    lines.append("## Workflow Steps")
    for step_trace in trace.step_traces:
        status_icon = {"success": "OK", "skipped": "SKIP", "failed": "FAIL"}.get(step_trace.status, "FAIL")
        lines.append(f"### Step {step_trace.step_number}: {step_trace.step_name}")
        lines.append(f"**Duration:** {step_trace.duration_seconds:.2f}s | **Status:** {status_icon}")
        lines.append("")

        if step_trace.input_data:
            lines.append("**Input:**")
            lines.append("```json")
            lines.append(_format_dict(step_trace.input_data))
            lines.append("```")
            lines.append("")

        if step_trace.output_data:
            lines.append("**Output:**")
            lines.append("```json")
            output_for_display = {}
            for k, v in step_trace.output_data.items():
                if isinstance(v, pd.DataFrame):
                    output_for_display[k] = f"DataFrame(shape={v.shape})"
                else:
                    output_for_display[k] = v
            lines.append(_format_dict(output_for_display))
            lines.append("```")
            lines.append("")

        if step_trace.product_changes:
            lines.append("**Product Focus:**")
            lines.append("```json")
            lines.append(_format_dict(step_trace.product_changes))
            lines.append("```")
            lines.append("")

    # Final Products
    lines.append("## Final Products")
    if trace.final_products:
        lines.append("| # | Item Name | Item IDs | Names in Message | Expected Action |")
        lines.append("|---|-----------|----------|------------------|-----------------|")
        for i, product in enumerate(trace.final_products, 1):
            item_nm = product.get('item_nm', 'Unknown')
            item_ids = ', '.join(str(x) for x in product.get('item_id', ['#']))
            names_in_msg = ', '.join(product.get('item_name_in_msg', []))
            actions = ', '.join(product.get('expected_action', ['Unknown']))
            lines.append(f"| {i} | {item_nm} | {item_ids} | {names_in_msg} | {actions} |")
    else:
        lines.append("*No products extracted*")

    return "\n".join(lines)
