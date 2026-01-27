#!/usr/bin/env python3
import requests
import time
import urllib.parse
import argparse
import importlib.util
import sys
from pathlib import Path

# Max retry count
MAX_RETRIES = 3


def load_config(config_path):
    """Load config from a Python file path"""
    config_path = Path(config_path).resolve()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    
    # Validate required attributes
    required_attrs = ["ZEPP_URL", "NOTEBOOK_ID", "PARAGRAPH_IDS_PRE", "PARAGRAPH_IDS"]
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise AttributeError(f"Config file must define '{attr}'")
    
    # PARAMS is optional if PARAMS_OUTER and PARAMS_INNER are provided
    if not hasattr(config, "PARAMS") and not (hasattr(config, "PARAMS_OUTER") and hasattr(config, "PARAMS_INNER")):
        raise AttributeError(f"Config file must define 'PARAMS' or both 'PARAMS_OUTER' and 'PARAMS_INNER'")
    
    # Set default values for optional attributes
    if not hasattr(config, "RESTART_SPARK_AT_START"):
        config.RESTART_SPARK_AT_START = True
    if not hasattr(config, "RESTART_SPARK_AT_END"):
        config.RESTART_SPARK_AT_END = True
    
    # Generate PARAMS from PARAMS_OUTER and PARAMS_INNER if provided
    if hasattr(config, "PARAMS_OUTER") and hasattr(config, "PARAMS_INNER"):
        if not hasattr(config, "PARAMS") or config.PARAMS is None:
            # Generate nested loop: OUTER x INNER
            # Each combination is a list of param strings
            outer_list = config.PARAMS_OUTER if config.PARAMS_OUTER else [[]]
            inner_list = config.PARAMS_INNER if config.PARAMS_INNER else [[]]
            
            # Convert to list format if needed
            if outer_list and isinstance(outer_list[0], str):
                outer_list = [[p] for p in outer_list]
            if inner_list and isinstance(inner_list[0], str):
                inner_list = [[p] for p in inner_list]
            
            # Generate combinations: outer x inner
            config.PARAMS = [
                outer + inner
                for outer in outer_list
                for inner in inner_list
            ]
            print(f"Generated {len(config.PARAMS)} parameter combinations from PARAMS_OUTER x PARAMS_INNER")
    
    # Ensure PARAMS exists
    if not hasattr(config, "PARAMS"):
        config.PARAMS = []
    
    return config


def restart_spark(zepp_url):
    """Restart Spark interpreter"""
    print("Restarting Spark interpreter...")
    url = f"{zepp_url}/api/interpreter/setting/restart/spark"
    requests.put(url)
    time.sleep(5)


def build_params_json(params):
    """Build JSON params string
    
    Args:
        params: List of "key:value" strings or list of such lists
                Examples:
                - ["suffix:c", "month:202512"] -> single execution with both params
                - [["suffix:c"], ["suffix:d"]] -> two executions with different params
    """
    if not params:
        return None
    
    # Handle list of strings (single execution with multiple params)
    if isinstance(params, list) and len(params) > 0 and isinstance(params[0], str):
        params_dict = {}
        for param in params:
            key, value = param.split(":", 1)
            params_dict[key] = urllib.parse.quote(value)
        return {"params": params_dict}
    
    # This shouldn't be called with nested lists anymore
    # (handled in main function)
    return None


def run_paragraph(zepp_url, notebook_id, paragraph_id, params=None):
    """Run a single paragraph and wait for completion. Returns True if success."""
    url = f"{zepp_url}/api/notebook/job/{notebook_id}/{paragraph_id}"
    
    print(f"Running: {paragraph_id}")
    
    if params:
        requests.post(url, json=params, headers={"Content-Type": "application/json"})
    else:
        requests.post(url, headers={"Content-Type": "application/json"})
    
    # Poll for completion
    status = "UNKNOWN"
    while status != "FINISHED":
        time.sleep(5)
        response = requests.get(url)
        data = response.json()
        body = data.get("body", {})
        status = body.get("status", "UNKNOWN")
        
        if status != "RUNNING":
            print(f"  Status: {status}")
        
        if status in ["ERROR", "ABORT"]:
            print(f"  FAILED: {status} in paragraph execution")
            
            # Extract and print error message
            error_messages = []
            
            # Try to get error from results
            results = body.get("results", {})
            if results.get("code") == "ERROR":
                result_msg = results.get("msg", [])
                if isinstance(result_msg, list):
                    for msg in result_msg:
                        if isinstance(msg, dict) and "data" in msg:
                            error_messages.append(msg["data"])
            
            # Try to get error from msg
            msg_list = body.get("msg", [])
            if isinstance(msg_list, list):
                for msg in msg_list:
                    if isinstance(msg, dict) and "data" in msg:
                        error_messages.append(msg["data"])
            
            # Print error messages
            if error_messages:
                print(f"\n{'='*80}")
                print(f"ERROR DETAILS:")
                print(f"{'='*80}")
                for idx, error_msg in enumerate(error_messages, 1):
                    # Limit error message length for readability
                    max_lines = 50
                    lines = error_msg.split('\n')
                    if len(lines) > max_lines:
                        truncated_msg = '\n'.join(lines[:max_lines]) + f"\n... (truncated, {len(lines) - max_lines} more lines)"
                    else:
                        truncated_msg = error_msg
                    print(f"\n--- Error Message {idx} ---")
                    print(truncated_msg)
                print(f"{'='*80}\n")
            
            return False
    
    print(f"  Completed: {paragraph_id}")
    return True


def run_pre_paragraphs(zepp_url, notebook_id, paragraph_ids_pre):
    """Run all pre-paragraphs. Returns True if all success."""
    for paragraph_id in paragraph_ids_pre:
        if not run_paragraph(zepp_url, notebook_id, paragraph_id):
            return False
    return True


def run_main_paragraphs_with_param(zepp_url, notebook_id, paragraph_ids, param_list):
    """Run all main paragraphs with given param list. Returns True if all success.
    
    Args:
        param_list: List of "key:value" strings or None
                   Examples: ["suffix:c"], ["suffix:c", "month:202512"], None
    """
    if param_list is not None:
        # Build params_json for this specific param combination
        params_json = build_params_json(param_list)
        print(f"\n=== Processing params: {param_list} ===")
    else:
        params_json = None
        print(f"\n=== Processing main paragraphs (no params) ===")
    
    for paragraph_id in paragraph_ids:
        if not run_paragraph(zepp_url, notebook_id, paragraph_id, params_json):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run Zeppelin paragraphs with configurable settings")
    parser.add_argument(
        "-c", "--config",
        default="config_raw_data.py",
        help="Path to config file (default: config_raw_data.py)"
    )
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    zepp_url = config.ZEPP_URL
    notebook_id = config.NOTEBOOK_ID
    paragraph_ids_pre = config.PARAGRAPH_IDS_PRE
    paragraph_ids = config.PARAGRAPH_IDS
    params = config.PARAMS
    restart_at_start = config.RESTART_SPARK_AT_START
    restart_at_end = config.RESTART_SPARK_AT_END
    
    print(f"Zeppelin URL: {zepp_url}")
    print(f"Notebook ID: {notebook_id}")
    print(f"Parameters: {params}")
    print(f"Restart Spark at start: {restart_at_start}")
    print(f"Restart Spark at end: {restart_at_end}\n")
    
    # Determine if params is list of lists or list of strings
    if params and len(params) > 0 and isinstance(params[0], list):
        # List of lists: [["suffix:c"], ["suffix:d"]] or [["suffix:c", "month:202512"], ...]
        params_to_run = params
    elif params and len(params) > 0 and isinstance(params[0], str):
        # List of strings: ["suffix:c", "suffix:d"] or ["suffix:c", "month:202512"]
        # Check if this should be treated as multiple executions or single execution
        # If all params have the same key, treat as multiple executions
        keys = [p.split(":", 1)[0] for p in params]
        if len(set(keys)) == 1:
            # All same key: ["suffix:c", "suffix:d"] -> multiple executions
            params_to_run = [[p] for p in params]
        else:
            # Different keys: ["suffix:c", "month:202512"] -> single execution
            params_to_run = [params]
    else:
        # Empty params
        params_to_run = [None]
    
    # Initial Spark restart (optional)
    if restart_at_start:
        restart_spark(zepp_url)
        print("Initial Spark restart completed\n")
    
    # Step 1: Run PRE paragraphs with retry (once for all params)
    print(f"\n{'='*80}")
    print(f"=== Running PRE paragraphs ===")
    print(f"{'='*80}")
    
    for attempt in range(MAX_RETRIES):
        print(f"\nPRE attempt {attempt + 1}/{MAX_RETRIES}")
        
        if run_pre_paragraphs(zepp_url, notebook_id, paragraph_ids_pre):
            print("PRE paragraphs completed successfully")
            break
        
        print(f"PRE paragraphs failed.")
    else:
        # All retry attempts exhausted for PRE
        print("\n" + "="*80)
        print("ERROR: PRE paragraphs failed after max retries. Exiting.")
        print("="*80)
        return
    
    # Step 2: Run MAIN paragraphs for each param with individual retry
    print(f"\n{'='*80}")
    print(f"=== Running MAIN paragraphs for {len(params_to_run)} parameter combination(s) ===")
    print(f"{'='*80}")
    
    failed_params = []
    for idx, param_list in enumerate(params_to_run, 1):
        print(f"\n{'='*80}")
        print(f"Processing parameter combination {idx}/{len(params_to_run)}")
        if param_list is not None:
            print(f"Parameters: {param_list}")
        print(f"{'='*80}")
        
        # Retry for this specific parameter combination
        success = False
        for attempt in range(MAX_RETRIES):
            print(f"\nAttempt {attempt + 1}/{MAX_RETRIES} for params {param_list}")
            
            if run_main_paragraphs_with_param(zepp_url, notebook_id, paragraph_ids, param_list):
                print(f"✓ Params {param_list} completed successfully")
                success = True
                break
            
            print(f"✗ Params {param_list} failed.")
        
        if not success:
            print(f"\n✗✗✗ ERROR: Params {param_list} failed after max retries.")
            failed_params.append(param_list)
        
        # Continue to next parameter combination even if this one failed
    
    # Summary
    print(f"\n{'='*80}")
    print(f"=== Execution Summary ===")
    print(f"{'='*80}")
    print(f"Total parameter combinations: {len(params_to_run)}")
    print(f"Successful: {len(params_to_run) - len(failed_params)}")
    print(f"Failed: {len(failed_params)}")
    
    if failed_params:
        print(f"\nFailed parameter combinations:")
        for param in failed_params:
            print(f"  - {param}")
        print("\nWARNING: Some parameter combinations failed. Check logs above.")
    else:
        print("\n✓ All parameter combinations completed successfully!")
    
    # Final Spark restart (optional)
    if restart_at_end:
        restart_spark(zepp_url)
        print("\nFinal Spark restart completed")
    
    print("\n" + "="*80)
    print("=== All done! ===")
    print("="*80)


if __name__ == "__main__":
    main()
