#!/usr/bin/env python3
"""Standalone entity extractor for model comparison.

Supports two backends:
  1. Anthropic API directly (claude-opus-4-6, claude-sonnet-4-5, etc.)
  2. Existing workflow models via LLMFactory/ChatOpenAI (ax, gpt, cld, gen, gem)

Usage:
    # Anthropic API models
    python apps/extract_entities_opus.py --batch-file tests/sample_30msgs.jsonl --model claude-opus-4-6
    python apps/extract_entities_opus.py --batch-file tests/sample_30msgs.jsonl --model claude-sonnet-4-5-20250514

    # Workflow models (via LLMFactory → ChatOpenAI → custom gateway)
    python apps/extract_entities_opus.py --batch-file tests/sample_30msgs.jsonl --model ax
    python apps/extract_entities_opus.py --batch-file tests/sample_30msgs.jsonl --model cld
    python apps/extract_entities_opus.py --batch-file tests/sample_30msgs.jsonl --model gpt
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory for package imports (same pattern as apps/cli.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv

DEFAULT_MODEL = "claude-opus-4-6"
MAX_TOKENS = 4096
TEMPERATURE = 0.0

# Workflow model aliases — routed through LLMFactory/ChatOpenAI
WORKFLOW_MODELS = {"ax", "gpt", "cld", "gen", "gem"}

# ---------------------------------------------------------------------------
# Custom prompt — designed from offer_master_data.csv domain knowledge
# ---------------------------------------------------------------------------
ENTITY_EXTRACTION_PROMPT = """\
# Task
SK텔레콤 MMS 광고 메시지에서 **핵심 오퍼링 엔티티**(Core Offering Entities)를 추출하라.
핵심 오퍼링이란 광고가 고객에게 제안하는 구체적인 상품·서비스·매장·이벤트를 의미한다.

# Entity Type Definitions (6 types)
아래 6개 타입 중 해당하는 것만 추출한다.

| Type | Code | 설명 | 예시 |
|------|------|------|------|
| **Store** | R | 물리적 대리점·매장 (지점명 포함) | CD대리점 동탄목동점, 유엔대리점 배곧사거리직영점, PS&M 동탄타임테라스점 |
| **Equipment** | E | 단말기·디바이스 모델명 | 아이폰 17, 갤럭시 Z 폴드7, iPad Air 13, 갤럭시 워치6 |
| **Product** | P | 요금제·부가서비스·유선상품 | 5GX 프라임 요금제, T끼리 온가족할인, 인터넷+IPTV, 로밍 baro 요금제 |
| **Subscription** | S | 월정액 구독 상품 | T 우주패스 올리브영&스타벅스&이마트24, T 우주패스 Netflix |
| **Voucher** | V | 제휴 할인·쿠폰·기프티콘 (브랜드+혜택 조합) | 도미노피자 50% 할인, 올리브영 3천 원 기프트카드, CGV 청년할인 |
| **Campaign** | X | 마케팅 캠페인·프로모션·이벤트명 | T Day, 0 day, special T, 고객 감사 패키지 |

# Extraction Rules

1. **Zero-Translation:** 원문에 등장하는 그대로 추출하라. 번역하지 말라.
   - 원문이 "아이폰 17 Pro"이면 → "아이폰 17 Pro" (NOT "iPhone 17 Pro")
   - 원문이 "T Day"이면 → "T Day" (NOT "티데이")

2. **Specificity:** 구체적인 고유명사만 추출하라. 포괄적 카테고리명은 제외한다.
   - ✅ "갤럭시 S25", "5GX 프라임 요금제", "CD대리점 동탄목동점"
   - ❌ "휴대폰", "요금제", "대리점", "인터넷"(단독), "할인"(단독)

3. **Store 추출:** 대리점명 + 지점명을 하나의 엔티티로 추출한다.
   - "유엔대리점 배곧사거리직영점" → 하나의 Store 엔티티

4. **Voucher 추출:** 제휴 브랜드 + 혜택 설명을 결합하여 추출한다.
   - "도미노피자 배달/방문 포장 50% 할인" → 하나의 Voucher 엔티티
   - 단, 브랜드만 언급되고 구체적 혜택이 없으면 추출하지 않는다.

5. **Strict Exclusions — 다음은 절대 추출하지 않는다:**
   - 할인 금액/비율 단독: "최대 22만원", "50% 할인", "25% 할인"
   - 일반 행위/설명: "매장 방문", "사전예약", "통신사 이동", "번호이동"
   - URL/연락처: "skt.sh/...", "1558", "1504"
   - 네비게이션: "바로 가기", "자세히 보기", "혜택받으러 가기"
   - 경쟁사 단독 언급: "KT", "LG U+", "알뜰폰" (비교 대상일 뿐)
   - 일반 용어: "5G", "LTE", "USIM" (단독, 상품명 아닌 경우)

# Output Format
반드시 아래 JSON 형식으로만 응답하라. JSON 외에 다른 텍스트를 포함하지 말라.

{
  "entities": [
    {"name": "엔티티명(원문 그대로)", "type": "R|E|P|S|V|X"}
  ]
}
"""


# --- Parsing helpers -----------------------------------------------------------

def parse_response(text: str) -> list[dict]:
    """Parse JSON response → list of entity dicts."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        return data.get("entities", [])
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


# --- LLM client abstraction ---------------------------------------------------

def create_anthropic_client():
    """Create Anthropic API client."""
    import anthropic
    return anthropic.Anthropic()


def create_workflow_llm(model_alias: str):
    """Create LLM via workflow LLMFactory (ChatOpenAI → custom gateway)."""
    from utils.llm_factory import LLMFactory
    factory = LLMFactory()
    return factory.create_model(model_alias)


def call_anthropic(client, message: str, model: str) -> tuple[str, dict]:
    """Call Anthropic API → (response_text, usage_dict)."""
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": message}],
    )
    return response.content[0].text, {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def call_workflow_llm(llm, message: str) -> tuple[str, dict]:
    """Call workflow LLM (ChatOpenAI) → (response_text, usage_dict)."""
    response = llm.invoke(message)
    usage = {}
    if hasattr(response, "response_metadata"):
        token_usage = response.response_metadata.get("token_usage", {})
        usage = {
            "input_tokens": token_usage.get("prompt_tokens", 0),
            "output_tokens": token_usage.get("completion_tokens", 0),
        }
    return response.content, usage


# --- Message loading -----------------------------------------------------------

def load_messages(args) -> list[dict]:
    """Return list of {'message_id': ..., 'message': ...}."""
    if args.message:
        return [{"message_id": "msg_0", "message": args.message}]

    path = Path(args.batch_file)
    if not path.exists():
        print(f"Error: batch file not found: {path}")
        sys.exit(1)

    messages = []
    with open(path, encoding="utf-8") as f:
        first_line = f.readline().strip()
        f.seek(0)

        is_jsonl = False
        if first_line.startswith("{"):
            try:
                json.loads(first_line)
                is_jsonl = True
            except json.JSONDecodeError:
                pass

        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if is_jsonl:
                rec = json.loads(line)
                messages.append({
                    "message_id": rec.get("message_id", f"msg_{i}"),
                    "message": rec["message"],
                })
            else:
                messages.append({"message_id": f"msg_{i}", "message": line})

    return messages


# --- Core extraction -----------------------------------------------------------

def extract(caller_fn, message: str) -> dict:
    """Call LLM via caller_fn and return parsed result dict."""
    user_content = f"{ENTITY_EXTRACTION_PROMPT}\n\n# Message\n{message}"

    t0 = time.time()
    raw, usage = caller_fn(user_content)
    elapsed = time.time() - t0

    entities = parse_response(raw)
    entity_names = [e["name"] for e in entities if "name" in e]
    entity_types = [f'{e["name"]}({e["type"]})' for e in entities if "name" in e and "type" in e]

    return {
        "extracted_entities": ", ".join(entity_names),
        "entity_types": ", ".join(entity_types),
        "raw_response": raw,
        "elapsed_s": round(elapsed, 2),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


# --- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract entities from MMS messages — supports Anthropic API and workflow models",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--message", help="Single message text")
    input_group.add_argument("--batch-file", help="Text file (one per line) or JSONL")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model: Anthropic API (claude-opus-4-6, etc.) or workflow alias (ax, cld, gpt, gen, gem). Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--output",
        help="Output CSV path (default: outputs/entities_{{model}}_{{timestamp}}.csv)",
    )
    args = parser.parse_args()

    model = args.model
    is_workflow = model in WORKFLOW_MODELS

    # Short label for filenames
    model_label = model.replace("claude-", "") if not is_workflow else model

    # Load .env from project root
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    # Initialize the appropriate backend
    if is_workflow:
        llm = create_workflow_llm(model)
        caller_fn = lambda msg: call_workflow_llm(llm, msg)
        backend = "workflow/ChatOpenAI"
    else:
        client = create_anthropic_client()
        caller_fn = lambda msg, m=model: call_anthropic(client, msg, m)
        backend = "anthropic"

    messages = load_messages(args)
    print(f"Model: {model} | Backend: {backend} | Messages: {len(messages)}")

    rows = []
    for i, rec in enumerate(messages):
        msg_id = rec["message_id"]
        msg = rec["message"]
        print(f"  [{i+1}/{len(messages)}] {msg_id}: {msg[:60]}...", end=" ", flush=True)
        try:
            result = extract(caller_fn, msg)
            n_ents = len(result["extracted_entities"].split(", ")) if result["extracted_entities"] else 0
            tok_str = f"{result['input_tokens']}+{result['output_tokens']} tokens" if result["input_tokens"] else ""
            print(f"→ {n_ents} entities ({result['elapsed_s']}s{', ' + tok_str if tok_str else ''})")
        except Exception as e:
            print(f"ERROR: {e}")
            result = {
                "extracted_entities": "",
                "entity_types": "",
                "raw_response": f"ERROR: {e}",
                "elapsed_s": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        rows.append({
            "message_id": msg_id,
            "model": model,
            "message": msg[:200],
            "extracted_entities": result["extracted_entities"],
            "entity_types": result["entity_types"],
            "raw_response": result["raw_response"],
            "correct_entities": "",
            "elapsed_s": result["elapsed_s"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
        })

    # Build output path
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(__file__).parent.parent / "outputs" / f"entities_{model_label}_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} rows → {out_path}")

    # Print summary table to stdout
    summary = df[["message_id", "entity_types", "elapsed_s"]].to_string(index=False)
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
