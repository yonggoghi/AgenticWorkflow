#!/usr/bin/env python3
"""
Entity Extraction Mode Comparison Evaluator
=============================================

Compares entity extraction quality across 4 context modes:
  - kg: Knowledge Graph extraction prompt
  - dag: DAG extraction prompt
  - ont: Ontology-based extraction prompt
  - typed (langextract): LangExtract engine with typed mode

Workflow:
  1. Generate ground truth using high-quality LLM (opus)
  2. Run entity extraction for each mode
  3. Calculate precision/recall/F1 per mode
  4. Rank modes by quality

Usage:
    # Full evaluation (ground truth + 4 modes + metrics)
    python tests/eval_entity_extraction_modes.py \
        --input outputs/entity_extraction_eval_anno.csv \
        --output-dir outputs/

    # Only generate ground truth annotations
    python tests/eval_entity_extraction_modes.py \
        --input outputs/entity_extraction_eval_anno.csv \
        --output-dir outputs/ \
        --only-annotate

    # Only compute metrics (if modes already extracted)
    python tests/eval_entity_extraction_modes.py \
        --input outputs/eval_with_modes.csv \
        --output-dir outputs/ \
        --only-metrics
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─── Ground Truth Generation ────────────────────────────────────────────

ANNOTATION_PROMPT = """당신은 SKT MMS 광고 메시지에서 핵심 엔티티를 추출하는 전문가입니다.

## 작업
아래 MMS 메시지를 분석하여 **offer/benefit 역할의 엔티티만** 추출하세요.
prerequisite(이미 보유한 것)는 제외합니다.

## 분석 프로세스 (반드시 순서대로 수행)

### Step 1: 메시지 이해
- 전체 메시지를 한 문장으로 요약
- 광고주의 의도 파악 (가입 유도? 구매 유도? 이용 안내? 혜택 안내?)

### Step 2: 가치 제안 식별
메시지가 제시하는 **모든 독립적 가치**를 파악:
- **즉각적 혜택**: 금전적 보상, 사은품, 할인 쿠폰
- **서비스 가치**: 제품/서비스 자체의 기능과 혜택
- **부가 혜택**: 추가 제공되는 가치 (있다면)

### Step 2.5: 역할 분류 (prerequisite vs offer)
각 엔티티의 역할을 판별:
- **prerequisite**: 타겟 고객이 **이미** 보유/가입/설치한 개체 (MMS 발송 대상 조건)
- **offer**: 메시지가 **새로** 구매/가입/사전예약/이용을 유도하는 개체
- **benefit**: prerequisite 위에서 활성화되는 새로운 기능/서비스/혜택

핵심 테스트: **"이 메시지가 해당 개체의 구매/가입/설치를 유도하는가?"**
→ YES → offer / NO (이미 보유를 전제) → prerequisite

prerequisite 판별 신호:
- "~님", "~고객", "~이용 중" → 이미 보유를 전제
- "~이용 안내" → 이미 보유한 것의 사용법 안내 → prerequisite
- 전이 규칙: prerequisite 번들을 통해 접근이 부여된 서비스도 prerequisite (예: "T우주 wavve"=prerequisite → "wavve"도 prerequisite)

**prerequisite는 최종 결과에서 제외합니다. offer와 benefit만 추출합니다.**

### Step 3: 엔티티 추출 (offer/benefit만, 대리점 제외)
식별된 가치 제안에서 **offer/benefit 역할**의 엔티티를 아래 7가지 타입에 따라 추출.
**대리점/매장(Store)은 추출하지 않는다** — main prompt에서 별도 추출.

1. **Product (단말기)**: 하드웨어 기기 모델명 — 아이폰 17, 갤럭시 Z 폴드7, 갤럭시 S25 울트라, 애플 워치, 에어팟
2. **RatePlan (요금제)**: 통신 요금제 — 5GX 프리미엄, 컴팩트 요금제, T 프라임 에센셜, 5GX 프라임(넷플릭스)
3. **Subscription (구독/부가서비스)**: 월정액 서비스, 앱 — T 우주패스, T 안심콜 라이트, 에이닷, 에이닷 전화, FLO 이용권, Netflix 광고형 스탠다드, 컬러링이용권, 콜키퍼, AI 안심 차단, T 청소년유해차단, T 오토 커넥트, T데이터 안심옵션
4. **Campaign (캠페인/이벤트)**: 마케팅 캠페인, 프로모션 — T Day, 0 day, special T, every day 혜택
5. **WiredService (유선 서비스)**: 인터넷/IPTV — 기가인터넷, B tv, B tv All, 기가라이트 와이파이
6. **PartnerBrand (제휴 브랜드)**: 프로모션의 핵심 주체인 제휴사 — 올리브영, 풀무원, 배달의민족, 하프클럽, 롯데월드
7. **Channel (접점 앱/플랫폼)**: 가입/이용 채널로서 언급된 앱 — T 월드, T 멤버십, T 멤버십 앱, PASS, 페이백쇼핑, ZEM 앱

### Step 4: 자기 검증
아래 기준으로 추출 결과를 검토하고, 누락이나 과잉이 있으면 수정:
- **offer/benefit 역할의 엔티티**가 빠짐없이 포함되었는가?
- **prerequisite(이미 보유)가 잘못 포함되지 않았는가?** (prerequisite는 제외해야 함)
- **제외 대상**에 해당하는 것이 잘못 포함되지 않았는가?

## 추출 규칙
1. **원문 그대로** 추출 (번역/축약/추론 금지)
2. **구체적 이름** 추출. 일반 범주("스마트폰", "요금제")가 아닌 구체적 명칭을 추출
3. 파생 관계가 있더라도 **각각 독립적으로** 추출 (예: "T 우주"와 "T 우주패스 Netflix"는 별개)
4. **메시지에 명시적으로 언급된 것만** 추출. 추론이나 배경지식으로 엔티티를 추가하지 않음
5. 중복 제거 후 가나다순 정렬

## 반드시 제외 (추출하지 않는 것)
- **대리점/매장명**: "CD대리점 동탄목동점", "에스알대리점 지행역점", "PS&M 동탄타임테라스점" 등 (main prompt에서 별도 추출)
- **금전적 혜택/할인 금액**: "20만 원 지원", "50% 할인", "캐시백", "기프티콘" (Benefit이지 엔티티가 아님)
- **고객센터/전화번호**: "SKT 고객센터(1558)", "114", "080-XXX"
- **URL/링크**: "https://...", "skt.sh/..."
- **수신거부 문구**: "무료 수신거부 1504"
- **네비게이션 라벨**: "바로 가기", "자세히 보기"
- **일반 기술 용어 단독**: "5G", "LTE", "USIM" (단, "5GX 프리미엄"처럼 상품명 일부면 추출)
- **일정/기간**: "2026년 2월 1일", "12월 31일까지"
- **타겟 고객 설명**: "VIP 고객님", "만 13~34세"
- **약정 조건**: "선택약정 24개월", "공시지원금"
- **법적 고지/유의사항**
- **"SKT", "SK텔레콤"** (발신자)
- **사은품 브랜드명 단독**: "[사죠영]", "[크레앙]"

## 경계 사례 가이드
| 메시지 내용 | 추출 여부 | 이유 |
|------------|----------|------|
| "T 멤버십 고객님께 풀무원 할인" | 풀무원 ✅ | T 멤버십=prerequisite("고객님께" → 이미 보유), 풀무원=PartnerBrand |
| "T Day 혜택: 올리브영 30% 할인" | T Day, 올리브영 ✅ | Campaign + PartnerBrand |
| "0 day 혜택 안내" | 0 day ✅ | Campaign |
| "PASS 앱에서 페이백쇼핑" | PASS, 페이백쇼핑 ✅ | Channel + Subscription |
| "에이닷 전화 이용 안내... AI 안심 차단 설정하면" | AI 안심 차단 ✅ (에이닷, 에이닷 전화는 prerequisite → 제외) | offer만 추출 |
| "5GX 요금제 혜택 안내... T 월드 앱에서 확인" | T 월드 ✅ (5GX는 이미 가입 prerequisite → 제외) | offer/channel만 추출 |
| "최대 22만 원 캐시백" | ❌ | Benefit (금전적 혜택) |
| "고객센터 114 문의" | ❌ | 연락처 |

## 잘못된 추출 예시
메시지: "T 멤버십 T Day 혜택 안내. 올리브영 30% 할인 쿠폰을 드립니다."
- ❌ 잘못: `올리브영 30% 할인 쿠폰` (금액/할인율 포함 = Benefit)
- ✅ 올바름: `T Day | 올리브영` (T 멤버십은 "혜택 안내" → prerequisite, 제외)

메시지: "에이닷 전화 이용 안내. AI 안심 차단을 설정하고 보이스피싱을 예방하세요."
- ❌ 잘못: `AI 안심 차단 | 에이닷 | 에이닷 전화` (에이닷, 에이닷 전화는 "이용 안내" → prerequisite)
- ✅ 올바름: `AI 안심 차단` (offer만 추출, prerequisite 제외)

## 출력 형식
### 분석
[Step 1-2 결과를 1-2줄로 간단히]

### 엔티티
[" | " 구분자로 나열, 가나다순. 엔티티가 없으면 "없음"]

## 분석 대상 메시지
{message}

위 메시지를 분석하여 엔티티를 추출해주세요."""


def _extract_entity_section(raw_response: str) -> str:
    """Extract the entity list from the structured LLM response.

    The prompt produces a response with "### 분석" and "### 엔티티" sections.
    This function extracts just the entity list from the "### 엔티티" section.
    Falls back to the full response if no section header is found.
    """
    # Look for "### 엔티티" header (with optional whitespace/markdown)
    match = re.search(r'###\s*엔티티\s*\n(.+)', raw_response, re.DOTALL)
    if match:
        entity_text = match.group(1).strip()
    else:
        # Fallback: try "엔티티" without ### prefix
        match = re.search(r'엔티티\s*[:：]\s*\n?(.+)', raw_response, re.DOTALL)
        if match:
            entity_text = match.group(1).strip()
        else:
            # No section found — use the whole response as-is
            entity_text = raw_response.strip()

    # Clean up markdown formatting
    entity_text = entity_text.replace('```', '').strip()
    entity_text = entity_text.strip('"').strip("'")
    # Take only the first line (entity list), ignore any trailing explanation
    first_line = entity_text.split('\n')[0].strip()
    return first_line


def generate_ground_truth(messages: List[str], batch_size: int = 5) -> List[str]:
    """Generate ground truth entity annotations using opus LLM."""
    from utils.llm_factory import LLMFactory

    factory = LLMFactory()
    llm = factory.create_model('opus')

    annotations = []
    total = len(messages)

    for i, msg in enumerate(messages):
        logger.info(f"[{i+1}/{total}] Annotating with opus...")
        prompt = ANNOTATION_PROMPT.format(message=msg)

        try:
            response = llm.invoke(prompt)
            raw = response.content.strip()
            # Extract only the "### 엔티티" section from the structured response
            annotation = _extract_entity_section(raw)
            logger.info(f"  -> {annotation[:100]}...")
            annotations.append(annotation)
        except Exception as e:
            logger.error(f"  -> Error: {e}")
            annotations.append("")

        # Brief pause to avoid rate limiting
        if (i + 1) % batch_size == 0 and i + 1 < total:
            logger.info(f"  Batch pause...")
            time.sleep(1)

    return annotations


# ─── Mode Extraction ────────────────────────────────────────────────────

MODE_CONFIGS = {
    'kg': {
        'entity_extraction_context_mode': 'kg',
        'extraction_engine': 'default',
    },
    'dag': {
        'entity_extraction_context_mode': 'dag',
        'extraction_engine': 'default',
    },
    'ont': {
        'entity_extraction_context_mode': 'ont',
        'extraction_engine': 'default',
    },
    'langextract': {
        'entity_extraction_context_mode': 'typed',
        'extraction_engine': 'langextract',
    },
}


def extract_entities_for_mode(
    messages: List[str],
    mode: str,
    llm_model: str = 'ax'
) -> List[str]:
    """Run entity extraction for a specific mode on all messages."""
    from tests.trace_product_extraction import ProductExtractionTracer

    config = MODE_CONFIGS[mode]

    extractor_kwargs = {
        'llm_model': llm_model,
        'entity_llm_model': llm_model,
        'entity_extraction_mode': 'llm',
        'offer_info_data_src': 'local',
        'product_info_extraction_mode': 'llm',
        'extract_entity_dag': False,
        'entity_extraction_context_mode': config['entity_extraction_context_mode'],
        'extraction_engine': config['extraction_engine'],
    }

    logger.info(f"Initializing tracer for mode '{mode}': {extractor_kwargs}")
    tracer = ProductExtractionTracer(extractor_kwargs)

    results = []
    total = len(messages)

    for i, msg in enumerate(messages):
        logger.info(f"[{i+1}/{total}] Mode={mode}: extracting...")
        try:
            trace = tracer.trace_message(msg, f"eval_{mode}_{i+1}")
            extracted = tracer._entity_trace.first_stage_entities or []
            sorted_extracted = sorted(extracted)
            result_str = " | ".join(sorted_extracted)
            logger.info(f"  -> {result_str[:100]}...")
            results.append(result_str)
        except Exception as e:
            logger.error(f"  -> Error: {e}")
            results.append("")

    return results


# ─── Metrics Calculation ────────────────────────────────────────────────

def normalize_entity(entity: str) -> str:
    """Normalize entity string for fuzzy matching."""
    s = entity.strip()
    # Remove common noise
    s = re.sub(r'\s+', ' ', s)  # normalize whitespace
    s = s.replace('(#)', '')     # remove unlinked marker
    s = s.strip()
    return s


def parse_entities(entities_str: str) -> Set[str]:
    """Parse pipe-separated entity string into a set."""
    if not entities_str or pd.isna(entities_str) or entities_str.strip() in ('', '없음'):
        return set()
    return {normalize_entity(e) for e in entities_str.split('|') if e.strip()}


def fuzzy_match_entity(pred: str, gold_set: Set[str], threshold: float = 0.7) -> bool:
    """Check if a predicted entity fuzzy-matches any gold entity."""
    pred_lower = pred.lower()
    for gold in gold_set:
        gold_lower = gold.lower()
        # Exact match
        if pred_lower == gold_lower:
            return True
        # Containment match (one contains the other)
        if pred_lower in gold_lower or gold_lower in pred_lower:
            return True
        # Token overlap
        pred_tokens = set(pred_lower.split())
        gold_tokens = set(gold_lower.split())
        if pred_tokens and gold_tokens:
            overlap = len(pred_tokens & gold_tokens)
            union = len(pred_tokens | gold_tokens)
            if union > 0 and overlap / union >= threshold:
                return True
    return False


def calculate_metrics(
    predictions: List[str],
    ground_truths: List[str],
    use_fuzzy: bool = True
) -> Dict[str, float]:
    """
    Calculate precision, recall, F1 for entity extraction.

    Args:
        predictions: List of pipe-separated predicted entity strings
        ground_truths: List of pipe-separated ground truth entity strings
        use_fuzzy: Use fuzzy matching instead of exact match

    Returns:
        Dict with precision, recall, f1, and per-message details
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_message = []

    for i, (pred_str, gold_str) in enumerate(zip(predictions, ground_truths)):
        pred_set = parse_entities(pred_str)
        gold_set = parse_entities(gold_str)

        if not gold_set:
            # No ground truth — skip this message
            per_message.append({
                'index': i,
                'precision': None, 'recall': None, 'f1': None,
                'tp': 0, 'fp': len(pred_set), 'fn': 0,
                'pred': pred_set, 'gold': gold_set,
                'skipped': True
            })
            continue

        # Count TP, FP
        tp = 0
        matched_golds = set()
        for pred in pred_set:
            if use_fuzzy:
                # Find best matching gold entity
                unmatched_golds = gold_set - matched_golds
                if fuzzy_match_entity(pred, unmatched_golds):
                    tp += 1
                    # Find and mark the matched gold
                    for g in unmatched_golds:
                        if fuzzy_match_entity(pred, {g}):
                            matched_golds.add(g)
                            break
            else:
                if pred in gold_set and pred not in matched_golds:
                    tp += 1
                    matched_golds.add(pred)

        fp = len(pred_set) - tp
        fn = len(gold_set) - len(matched_golds)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_message.append({
            'index': i,
            'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn,
            'pred': pred_set, 'gold': gold_set,
            'extra': pred_set - matched_golds if not use_fuzzy else None,
            'missing': gold_set - matched_golds,
            'skipped': False
        })

    # Micro-averaged metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Macro-averaged metrics (excluding skipped)
    valid = [m for m in per_message if not m['skipped']]
    macro_precision = sum(m['precision'] for m in valid) / len(valid) if valid else 0.0
    macro_recall = sum(m['recall'] for m in valid) / len(valid) if valid else 0.0
    macro_f1 = sum(m['f1'] for m in valid) / len(valid) if valid else 0.0

    return {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'num_evaluated': len(valid),
        'num_skipped': len(per_message) - len(valid),
        'per_message': per_message
    }


# ─── Report Generation ──────────────────────────────────────────────────

def print_comparison_report(mode_metrics: Dict[str, Dict], output_file: str = None):
    """Print a formatted comparison report and optionally save to file."""
    lines = []
    lines.append("=" * 80)
    lines.append("Entity Extraction Mode Comparison Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # Summary table
    lines.append("")
    lines.append(f"{'Mode':<15} {'Micro-P':>8} {'Micro-R':>8} {'Micro-F1':>9} {'Macro-P':>8} {'Macro-R':>8} {'Macro-F1':>9} {'TP':>5} {'FP':>5} {'FN':>5}")
    lines.append("-" * 95)

    # Sort by micro F1 descending
    sorted_modes = sorted(mode_metrics.items(), key=lambda x: x[1]['micro_f1'], reverse=True)

    for rank, (mode, m) in enumerate(sorted_modes, 1):
        lines.append(
            f"#{rank} {mode:<12} {m['micro_precision']:>7.1%} {m['micro_recall']:>7.1%} {m['micro_f1']:>8.1%}"
            f" {m['macro_precision']:>7.1%} {m['macro_recall']:>7.1%} {m['macro_f1']:>8.1%}"
            f" {m['total_tp']:>5} {m['total_fp']:>5} {m['total_fn']:>5}"
        )

    lines.append("-" * 95)
    lines.append("")

    # Ranking
    lines.append("## Final Ranking (by Micro-F1)")
    for rank, (mode, m) in enumerate(sorted_modes, 1):
        medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, '  ')
        lines.append(f"  {medal} #{rank}: {mode} (F1={m['micro_f1']:.1%}, P={m['micro_precision']:.1%}, R={m['micro_recall']:.1%})")
    lines.append("")

    # Per-mode error analysis (top 5 worst messages)
    for mode, m in sorted_modes:
        worst = sorted(
            [msg for msg in m['per_message'] if not msg['skipped']],
            key=lambda x: x['f1']
        )[:5]

        lines.append(f"### {mode} - Worst 5 messages")
        for msg in worst:
            lines.append(f"  msg[{msg['index']}]: F1={msg['f1']:.2f} (TP={msg['tp']}, FP={msg['fp']}, FN={msg['fn']})")
            if msg.get('missing'):
                lines.append(f"    Missing: {', '.join(sorted(msg['missing']))}")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to {output_file}")

    return report


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entity Extraction Mode Comparison Evaluator")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--output-dir", "-o", default="outputs/", help="Output directory")
    parser.add_argument("--llm-model", "-m", default="ax", help="LLM model for extraction (default: ax)")
    parser.add_argument("--only-annotate", action="store_true", help="Only generate ground truth, skip extraction")
    parser.add_argument("--only-metrics", action="store_true", help="Only compute metrics from existing data")
    parser.add_argument("--modes", nargs='+', default=['kg', 'dag', 'ont', 'langextract'],
                        choices=['kg', 'dag', 'ont', 'langextract'],
                        help="Modes to evaluate (default: all 4)")
    parser.add_argument("--fuzzy", action="store_true", default=True, help="Use fuzzy matching (default)")
    parser.add_argument("--no-fuzzy", dest="fuzzy", action="store_false", help="Use exact matching only")

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load input
    df = pd.read_csv(args.input, encoding='utf-8-sig')
    messages = df['mms'].tolist()
    logger.info(f"Loaded {len(messages)} messages from {args.input}")

    if args.only_metrics:
        # ─── Metrics only ───
        logger.info("Computing metrics only...")
        gt_col = 'correct_extracted_entities'
        if gt_col not in df.columns or df[gt_col].isna().all():
            logger.error(f"No ground truth found in '{gt_col}' column!")
            sys.exit(1)

        ground_truths = df[gt_col].fillna('').tolist()
        mode_metrics = {}

        # Auto-detect all extracted_entities_* columns
        prefix = 'extracted_entities_'
        extract_cols = [c for c in df.columns if c.startswith(prefix) and df[c].notna().any()]
        for col in extract_cols:
            mode_label = col[len(prefix):]  # e.g. "ax", "cld", "ax_dag", "cld_dag"
            predictions = df[col].fillna('').tolist()
            metrics = calculate_metrics(predictions, ground_truths, use_fuzzy=args.fuzzy)
            mode_metrics[mode_label] = metrics
            logger.info(f"Mode '{mode_label}': F1={metrics['micro_f1']:.3f}")

        report_file = output_path / f"eval_report_{timestamp}.txt"
        print_comparison_report(mode_metrics, str(report_file))
        return

    # ─── Step 1: Ground truth annotation ───
    gt_col = 'correct_extracted_entities'
    has_gt = gt_col in df.columns and df[gt_col].notna().any() and (df[gt_col] != '').any()

    if not has_gt:
        logger.info("=" * 60)
        logger.info("Step 1: Generating ground truth with opus LLM...")
        logger.info("=" * 60)
        annotations = generate_ground_truth(messages)
        df[gt_col] = annotations
        # Save intermediate result
        gt_file = output_path / f"eval_annotated_{timestamp}.csv"
        df.to_csv(gt_file, index=False, encoding='utf-8-sig')
        logger.info(f"Ground truth saved to {gt_file}")
    else:
        logger.info("Ground truth already exists, skipping annotation.")
        gt_file = args.input

    if args.only_annotate:
        logger.info("Done (--only-annotate). Review annotations and re-run without flag.")
        return

    # ─── Step 2: Extract entities for each mode ───
    ground_truths = df[gt_col].fillna('').tolist()
    mode_metrics = {}

    for mode in args.modes:
        col = f'extracted_entities_{args.llm_model}_{mode}'

        # Check if this mode was already extracted
        if col in df.columns and df[col].notna().any() and (df[col] != '').any():
            logger.info(f"Mode '{mode}' already extracted (column '{col}'), using existing data.")
            predictions = df[col].fillna('').tolist()
        else:
            logger.info("=" * 60)
            logger.info(f"Step 2: Extracting entities for mode '{mode}'...")
            logger.info("=" * 60)

            predictions = extract_entities_for_mode(messages, mode, args.llm_model)
            df[col] = predictions

            # Save after each mode (intermediate checkpoint)
            checkpoint_file = output_path / f"eval_checkpoint_{timestamp}.csv"
            df.to_csv(checkpoint_file, index=False, encoding='utf-8-sig')
            logger.info(f"Checkpoint saved: {checkpoint_file}")

        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truths, use_fuzzy=args.fuzzy)
        mode_metrics[mode] = metrics
        logger.info(f"Mode '{mode}': Micro-F1={metrics['micro_f1']:.3f}, P={metrics['micro_precision']:.3f}, R={metrics['micro_recall']:.3f}")

    # ─── Step 3: Final report ───
    # Save final CSV
    final_file = output_path / f"eval_complete_{timestamp}.csv"
    df.to_csv(final_file, index=False, encoding='utf-8-sig')
    logger.info(f"Final results saved to {final_file}")

    # Print comparison report
    report_file = output_path / f"eval_report_{timestamp}.txt"
    print_comparison_report(mode_metrics, str(report_file))


if __name__ == "__main__":
    main()
