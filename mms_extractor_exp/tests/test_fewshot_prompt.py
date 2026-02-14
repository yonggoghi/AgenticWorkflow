"""
Test: Compare old schema-based prompt vs new few-shot prompt.

Runs the revised few-shot prompt through the LLM and displays the result.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")


def run_test(model_id: str = "ax"):
    from prompts.main_extraction_prompt import build_extraction_prompt
    from utils.llm_factory import LLMFactory
    from utils.json_utils import extract_json_objects

    factory = LLMFactory()
    llm = factory.create_model(model_id)

    test_message = (
        "[SKT] T 우주패스 쇼핑 출시! "
        "지금 링크를 눌러 가입하면 첫 달 1,000원에 이용 가능합니다. "
        "가입 고객 전원에게 11번가 포인트 3,000P와 아마존 무료배송 쿠폰을 드립니다. "
        "문의: 114"
    )

    rag_context = (
        "### 참고용 후보 상품 이름 목록 ###\n"
        "- T 우주패스 쇼핑\n"
        "- T 우주패스 올리브영&스타벅스\n"
        "- 11번가 스마트배송\n"
    )

    # Build prompt (new few-shot version)
    prompt = build_extraction_prompt(
        message=test_message,
        rag_context=rag_context,
        product_info_extraction_mode='llm',
    )

    print("=" * 70)
    print(f"MODEL: {model_id}")
    print(f"PROMPT LENGTH: {len(prompt)} chars (~{len(prompt)//3} tokens)")
    print(f"MESSAGE: {test_message}")
    print("=" * 70)

    # Call LLM
    print("\n▶ Calling LLM...")
    t0 = time.time()
    response = llm.invoke(prompt)
    elapsed = time.time() - t0
    result_text = response.content if hasattr(response, 'content') else str(response)
    print(f"  ⏱  {elapsed:.1f}s")

    # Parse JSON
    print(f"\n▶ Raw response ({len(result_text)} chars):")
    print(result_text)

    json_objects_list = extract_json_objects(result_text)
    if json_objects_list:
        parsed = json_objects_list[-1]
        print(f"\n▶ Parsed JSON:")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))

        # Check for schema response
        from utils.validation_utils import detect_schema_response
        is_schema = detect_schema_response(parsed)
        print(f"\n▶ Schema response detected: {is_schema}")
    else:
        print("\n▶ FAILED to parse JSON from response!")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "ax"
    run_test(model)
