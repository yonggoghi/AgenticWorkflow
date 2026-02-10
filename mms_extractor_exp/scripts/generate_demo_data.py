#!/usr/bin/env python3
"""
Demo Data Generator for Presentation App

Runs the full MMS extraction pipeline on sample messages and saves
results + DAG images as JSON files for offline demo use.

Usage:
    python scripts/generate_demo_data.py
"""

import json
import sys
import os
import time
import re
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.mms_extractor import MMSExtractor, process_message_worker
from core.workflow_core import WorkflowState
from utils.hash_utils import sha256_hash

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "data" / "demo_results"
DAG_IMAGES_DIR = PROJECT_ROOT / "dag_images"

# Sample messages (same as apps/demo_streamlit.py)
SAMPLE_MESSAGES = [
    {
        "title": "10월 0 day 혜택 (명랑핫도그)",
        "content": "(광고)[SKT] 10월 0 day 혜택 안내__<10월 30일(목) 혜택>_만 13~34세 고객님이라면_SKT 0 day_[명랑핫도그 500원에 드림]_(선착순 5천 명)__▶ 자세히 보기: https://t-mms.kr/t.do?m=#61&s=34168&a=&u=https://bit.ly/46k1M7H__■ 문의: SKT 고객센터(1558, 무료)__무료 수신거부 1504"
    },
    {
        "title": "큰사랑대리점 갤럭시 시리즈 특가",
        "content": "(광고)[SKT] 큰사랑대리점 구래직영점 12월 혜택 안내__고객님, 안녕하세요._큰사랑대리점 구래직영점에서 12월 혜택을 안내드립니다._갤럭시 시리즈 특가 행사와 다양한 추가 할인 혜택을 만나보세요.__■ 갤럭시 시리즈 특가 행사_- 갤럭시 S25 할부 원금 40만 원 대_- 갤럭시 Z 폴드7 할부 원금 100만 원 대_- 갤럭시 S25 엣지 할부 원금 60만 원 대_- 통신사 번호이동 시 갤럭시 A17 추가 지원금 포함 할부 원금 혜택__■ 할부 원금 추가 할인 혜택_- 제휴 카드 이용 시 추가 할인_- 인터넷과 TV 동시 신청 시 할부 원금 추가 할인_- 타사 가"
    },
    {
        "title": "T 우주 YouTube Premium 구독 안내",
        "content": "e One 구독 안내__#04 고객님, 안녕하세요.__T 우주 with YouTube Premium_12월 1일(월) 가격 인상 전 마지막 기회!__YouTube Premium과 Google One 100GB를 월 13,900원에 구독해 보세요.__▶ 구독하기: https://t-mms.kr/asg/#74__■ 문의: T 우주 고객센터(1505, 무료)__구독 마켓, T 우주__무료 수신거부 1504"
    },
]


def sanitize_filename(title: str) -> str:
    """Create a safe filename from a title."""
    # Remove special characters, keep Korean and alphanumeric
    sanitized = re.sub(r'[^\w가-힣]', '_', title)
    # Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return sanitized[:50]


def run_pipeline_with_timing(extractor, message: str, message_id: str = '#'):
    """
    Run pipeline step-by-step and capture per-step timings.
    Returns (result_dict, step_timings, total_duration).
    """
    initial_state = WorkflowState(
        mms_msg=message,
        extractor=extractor,
        message_id=message_id
    )

    steps = extractor.workflow_engine.steps
    state = initial_state
    step_timings = []
    total_start = time.time()

    for i, step in enumerate(steps, 1):
        step_name = step.name()
        step_start = time.time()

        if not step.should_execute(state):
            step_duration = time.time() - step_start
            step_timings.append({
                "step": step_name,
                "number": i,
                "duration": round(step_duration, 4),
                "status": "skipped"
            })
            state.add_history(step_name, step_duration, "skipped")
            continue

        try:
            state = step.execute(state)
            status = "success" if not state.has_error() else "failed"
        except Exception as e:
            status = "failed"
            print(f"  Step {step_name} failed: {e}")

        step_duration = time.time() - step_start
        step_timings.append({
            "step": step_name,
            "number": i,
            "duration": round(step_duration, 4),
            "status": status
        })

        if state.has_error():
            break

    total_duration = round(time.time() - total_start, 2)

    final_result = state.get("final_result", {})
    raw_result = state.get("raw_result", {})
    final_result['message_id'] = message_id
    raw_result['message_id'] = message_id

    # Capture intermediate data
    rag_context = state.get("rag_context", "")
    entities_from_kiwi = state.get("entities_from_kiwi", [])
    cand_item_list = state.get("cand_item_list", None)
    if cand_item_list is not None:
        if hasattr(cand_item_list, 'to_dict'):
            cand_items_json = cand_item_list.to_dict(orient='records')
        elif isinstance(cand_item_list, list):
            cand_items_json = cand_item_list
        else:
            cand_items_json = list(cand_item_list)
    else:
        cand_items_json = []

    return {
        "ext_result": final_result,
        "raw_result": raw_result,
        "rag_context": rag_context,
        "entities_from_kiwi": list(entities_from_kiwi) if entities_from_kiwi else [],
        "cand_item_list": cand_items_json,
    }, step_timings, total_duration


def generate_demo_data():
    """Generate demo data by running the pipeline on sample messages."""
    print("=" * 60)
    print("MMS Extractor Demo Data Generator")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DAG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    print("\n[1/2] Initializing MMSExtractor...")
    init_start = time.time()
    extractor = MMSExtractor(
        extract_entity_dag=True,
        llm_model='ax',
        offer_info_data_src='local',
    )
    init_duration = time.time() - init_start
    print(f"  Initialized in {init_duration:.1f}s")

    # Process each sample message
    print(f"\n[2/2] Processing {len(SAMPLE_MESSAGES)} sample messages...\n")

    generated_files = []

    for idx, sample in enumerate(SAMPLE_MESSAGES):
        title = sample["title"]
        message = sample["content"]
        # Hash the stripped message to match the pipeline (InputValidationStep strips whitespace)
        msg_hash = sha256_hash(message.strip())
        dag_filename = f"dag_#_{msg_hash}.png"

        print(f"--- Message {idx + 1}/{len(SAMPLE_MESSAGES)}: {title} ---")

        # Run pipeline with step-by-step timing
        print(f"  Running pipeline...")
        result, step_timings, total_duration = run_pipeline_with_timing(
            extractor, message, message_id='#'
        )

        # Print step timings
        for st in step_timings:
            icon = {"success": "OK", "skipped": "SKIP", "failed": "FAIL"}.get(st["status"], "?")
            print(f"    [{icon}] Step {st['number']:2d} {st['step']:<30s} {st['duration']:.3f}s")
        print(f"  Total: {total_duration:.2f}s")

        # Verify DAG image
        dag_path = DAG_IMAGES_DIR / dag_filename
        dag_exists = dag_path.exists()
        if dag_exists:
            print(f"  DAG image: {dag_filename} ({dag_path.stat().st_size:,} bytes)")
        else:
            print(f"  DAG image: {dag_filename} (NOT FOUND)")

        # Build JSON output
        demo_json = {
            "message": message,
            "title": title,
            "ext_result": result.get("ext_result", {}),
            "raw_result": result.get("raw_result", {}),
            "rag_context": result.get("rag_context", ""),
            "entities_from_kiwi": result.get("entities_from_kiwi", []),
            "cand_item_list": result.get("cand_item_list", []),
            "dag_image_filename": dag_filename if dag_exists else None,
            "step_timings": step_timings,
            "total_duration": total_duration,
            "metadata": {
                "llm_model": "ax",
                "mode": "llm",
                "generated_at": datetime.now().isoformat(),
                "msg_hash": msg_hash,
            }
        }

        # Save JSON
        safe_title = sanitize_filename(title)
        output_filename = f"{idx + 1}_{safe_title}.json"
        output_path = OUTPUT_DIR / output_filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(demo_json, f, ensure_ascii=False, indent=2, default=str)

        print(f"  Saved: {output_path.name}")
        generated_files.append(output_path)
        print()

    # Summary
    print("=" * 60)
    print("Generation Complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files generated: {len(generated_files)}")
    for f in generated_files:
        size = f.stat().st_size
        print(f"    - {f.name} ({size:,} bytes)")
    print("=" * 60)


if __name__ == "__main__":
    generate_demo_data()
