#!/usr/bin/env python3
"""
Program Classification Tracing Tool
=====================================

End-to-end trace of the ProgramClassificationStep (Step 3).
Shows how the message is embedded, compared against all program categories,
and which top-N candidates are selected — including similarity scores,
clue tags, and how the result flows into the RAG context (Step 4).

Usage:
    # Default: use hardcoded test message
    python tests/trace_program_classification.py

    # Custom message
    python tests/trace_program_classification.py --message "아이폰 17 구매하세요"

    # Change number of candidate programs
    python tests/trace_program_classification.py --num-cand-pgms 10

    # Show all programs (not just top-N)
    python tests/trace_program_classification.py --show-all
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress most logs during tracing
logging.basicConfig(
    level=logging.WARNING,
    format='%(name)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ─── Default test message ───────────────────────────────────────────────────
DEFAULT_MESSAGE = (
    "[SK텔레콤] 새서울대리점 대치직영점 10월 혜택 안내드립니다.\t"
    "(광고)[SKT] 새서울대리점 대치직영점 10월 혜택 안내__"
    "고객님, 안녕하세요._대치역 8번 출구 인근 새서울대리점 대치직영점에서 10월 혜택을 안내드립니다._"
    "특별 이벤트와 다양한 혜택을 경험해 보세요.__"
    "■ 갤럭시 Z 플립7/폴드7 구매 혜택_- 최대 할인 제공_- 갤럭시 워치 무료 증정(5GX 프라임 요금제 이용 시)__"
    "■ 아이폰 신제품 구매 혜택_- 최대 할인 및 쓰던 폰 반납 시 최대 보상 제공_"
    "- 아이폰 에어 구매 시 에어팟 증정(5GX 프라임 요금제 이용 시)__"
    "■ 공신폰/부모님폰 한정 수량 특별 할인_- 매일 선착순 3명 휴대폰 최대 할인__"
    "■ 새서울대리점 대치직영점_- 주소: 서울특별시 강남구 삼성로 151_"
    "- 연락처: 02-539-9965_"
    "- 찾아오시는 길: 3호선 대치역 8번 출구에서 직진 50m, 선경아파트 상가 bbq \\ 건물 1층_"
    "- 영업 시간: 평일 오전 10시 30분~오후 7시, 토요일 오전 11시~오후 6시__"
    "▶ 매장 홈페이지 예약/상담 : https://t-mms.kr/t.do?m=#61&s=34192&a=&u=https://tworldfriends.co.kr/D138580279__"
    "■ 문의: SKT 고객센터(1558, 무료)__"
    "SKT와 함께해 주셔서 감사합니다.__무료 수신거부 1504"
)


def print_header(title: str, width: int = 70):
    print(f"\n{'=' * width}")
    print(f" {title}")
    print(f"{'=' * width}")


def print_section(title: str, width: int = 70):
    print(f"\n{'─' * width}")
    print(f" {title}")
    print(f"{'─' * width}")


def trace_program_classification(message: str, num_cand_pgms: int = 20,
                                  num_select_pgms: int = None,
                                  show_all: bool = False,
                                  data_source: str = 'local'):
    """Run end-to-end program classification trace."""
    from core.mms_extractor import MMSExtractor

    # ── Step 0: Initialize ───────────────────────────────────────────────
    print_header("Program Classification Trace")
    print(f"Message length: {len(message)} chars")
    print(f"Message preview: {message[:120]}...")
    print(f"num_cand_pgms: {num_cand_pgms}")
    print(f"data_source: {data_source}")

    print_section("Initializing MMSExtractor")
    t0 = time.time()
    extractor = MMSExtractor(
        offer_info_data_src=data_source,
    )
    # Override num_cand_pgms if specified
    extractor.num_cand_pgms = num_cand_pgms
    if num_select_pgms is not None:
        extractor.num_select_pgms = num_select_pgms
    init_time = time.time() - t0
    print(f"Initialized in {init_time:.1f}s")
    print(f"Config num_cand_pgms overridden to: {num_cand_pgms}")
    print(f"Config num_select_pgms: {extractor.num_select_pgms}")

    # ── Program data overview ────────────────────────────────────────────
    print_section("Program Data Overview")
    pgm_pdf = extractor.pgm_pdf
    print(f"Total programs: {len(pgm_pdf)}")
    print(f"Columns: {list(pgm_pdf.columns)}")
    if not pgm_pdf.empty:
        print(f"Sample (first 3):")
        for _, row in pgm_pdf.head(3).iterrows():
            pgm_nm = row.get('pgm_nm', 'N/A')
            clue_tag = row.get('clue_tag', 'N/A')
            print(f"  - {pgm_nm} | clue: {clue_tag[:80]}")

    # ── Embedding model info ─────────────────────────────────────────────
    print_section("Embedding Model Info")
    emb_model = extractor.emb_model
    clue_embeddings = extractor.clue_embeddings
    if emb_model is not None:
        model_name = getattr(emb_model, 'model_name', getattr(emb_model, '_model_name_or_path', str(type(emb_model))))
        print(f"Embedding model: {model_name}")
    else:
        print("Embedding model: None")
    print(f"Clue embeddings shape: {clue_embeddings.shape}")

    # ── Step 1: Input Validation (lightweight) ───────────────────────────
    print_section("Step 1: Input Validation")
    from utils import preprocess_text
    msg = preprocess_text(message)
    print(f"Preprocessed message length: {len(msg)} chars")
    print(f"Preprocessed preview: {msg[:150]}...")

    # ── Step 3: Program Classification (core trace) ──────────────────────
    print_section("Step 3: Program Classification")

    import torch

    # 3a: Message embedding
    t0 = time.time()
    mms_embedding = emb_model.encode(
        [msg.lower()], convert_to_tensor=True, show_progress_bar=False
    )
    emb_time = time.time() - t0
    print(f"Message embedding: shape={mms_embedding.shape}, time={emb_time:.3f}s")

    # 3b: Cosine similarity against all programs
    t0 = time.time()
    similarities = torch.nn.functional.cosine_similarity(
        mms_embedding, clue_embeddings, dim=1
    ).cpu().numpy()
    sim_time = time.time() - t0
    print(f"Cosine similarity: computed {len(similarities)} scores in {sim_time:.4f}s")

    # 3c: Sort and rank
    import re
    pgm_pdf_tmp = pgm_pdf.copy()
    pgm_pdf_tmp['sim'] = similarities
    pgm_pdf_tmp = pgm_pdf_tmp.sort_values('sim', ascending=False).reset_index(drop=True)

    # 3d: Display top-N candidates
    print_section(f"Top-{num_cand_pgms} Candidate Programs")
    top_n = pgm_pdf_tmp.head(num_cand_pgms)
    for idx, row in top_n.iterrows():
        pgm_nm = row.get('pgm_nm', 'N/A')
        pgm_nm_clean = re.sub(r'\[.*?\]', '', pgm_nm)
        clue_tag = row.get('clue_tag', 'N/A')
        sim = row.get('sim', 0)
        pgm_id = row.get('pgm_id', 'N/A')
        print(f"  [{idx+1}] sim={sim:.4f} | {pgm_nm_clean}")
        print(f"       clue: {clue_tag[:100]}")
        if pgm_id != 'N/A':
            print(f"       pgm_id: {pgm_id}")

    # 3e: pgm_cand_info string (what goes into RAG context)
    pgm_cand_info = "\n\t".join(
        top_n[['pgm_nm', 'clue_tag']].apply(
            lambda x: re.sub(r'\[.*?\]', '', x['pgm_nm']) + " : " + x['clue_tag'], axis=1
        ).to_list()
    )

    print_section("Generated pgm_cand_info (for RAG context)")
    print(pgm_cand_info)

    # ── Step 4: Context Preparation (how pgm_info becomes RAG context) ──
    print_section("Step 4: RAG Context Integration")
    num_select_pgms = extractor.num_select_pgms
    if num_cand_pgms > 0:
        rag_context = (
            f"\n### 광고 분류 기준 정보 (pgm 후보 목록) ###\n"
            f"For the pgm field, select up to {num_select_pgms} from the following list. Copy the name EXACTLY.\n"
            f"\t{pgm_cand_info}"
        )
        print(f"RAG context length: {len(rag_context)} chars")
        print(f"num_select_pgms: {num_select_pgms}")
        print(f"RAG context:\n{rag_context}")
    else:
        print("num_cand_pgms=0, no program context added to RAG")

    # ── Step 5: Full LLM Extraction Prompt (with pgm candidates) ───────
    print_section("Step 5: LLM Extraction Prompt (with pgm candidates)")
    from prompts.main_extraction_prompt import build_extraction_prompt

    # Build the prompt as the pipeline would (LLM mode, pgm-only RAG context)
    full_prompt = build_extraction_prompt(
        message=msg,
        rag_context=rag_context if num_cand_pgms > 0 else "",
        product_element=None,
        product_info_extraction_mode=extractor.product_info_extraction_mode,
        num_select_pgms=num_select_pgms,
    )
    print(f"Prompt length: {len(full_prompt)} chars (~{len(full_prompt)//3} tokens)")
    print(f"Product info extraction mode: {extractor.product_info_extraction_mode}")
    print(f"\n{'·' * 70}")
    print(full_prompt)
    print(f"{'·' * 70}")

    # ── Similarity distribution ──────────────────────────────────────────
    print_section("Similarity Distribution")
    print(f"Max:    {similarities.max():.4f}")
    print(f"Min:    {similarities.min():.4f}")
    print(f"Mean:   {similarities.mean():.4f}")
    print(f"Median: {float(pd.Series(similarities).median()):.4f}")
    print(f"Std:    {similarities.std():.4f}")

    # Histogram-like distribution
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    counts, _ = pd.cut(similarities, bins=bins, retbins=True)
    dist = counts.value_counts().sort_index()
    print("\nDistribution:")
    for interval, count in dist.items():
        bar = '#' * min(count, 50)
        print(f"  {interval}: {count:3d} {bar}")

    # ── Full ranking (optional) ──────────────────────────────────────────
    if show_all:
        print_section(f"Full Program Ranking ({len(pgm_pdf_tmp)} programs)")
        for idx, row in pgm_pdf_tmp.iterrows():
            pgm_nm = re.sub(r'\[.*?\]', '', row.get('pgm_nm', 'N/A'))
            sim = row.get('sim', 0)
            print(f"  [{idx+1:3d}] sim={sim:.4f} | {pgm_nm}")

    # ── Full pipeline run (optional comparison) ──────────────────────────
    print_section("Full Pipeline Execution (for comparison)")
    t0 = time.time()
    result = extractor.process_message(message)
    pipeline_time = time.time() - t0
    print(f"Pipeline completed in {pipeline_time:.1f}s")

    ext_result = result.get('ext_result', {})
    raw_result = result.get('raw_result', {})

    print(f"\nExtracted pgm (raw_result): {raw_result.get('pgm', [])}")
    print(f"Extracted pgm (ext_result): {ext_result.get('pgm', [])}")
    print(f"Extracted pgm_id: {ext_result.get('pgm_id', [])}")
    print(f"Extracted title: {raw_result.get('title', 'N/A')}")
    print(f"Extracted purpose: {raw_result.get('purpose', [])}")

    products = raw_result.get('product', [])
    if products:
        print(f"Extracted products ({len(products)}):")
        for p in products:
            print(f"  - {p.get('name', 'N/A')} ({p.get('action', 'N/A')})")

    channels = raw_result.get('channel', [])
    if channels:
        print(f"Extracted channels ({len(channels)}):")
        for c in channels:
            print(f"  - {c.get('type', 'N/A')}: {c.get('value', 'N/A')} ({c.get('action', 'N/A')})")

    print_header("Trace Complete")


def main():
    parser = argparse.ArgumentParser(description='Trace Program Classification Step')
    parser.add_argument('--message', '-m', type=str, default=None,
                        help='MMS message to trace (default: hardcoded test message)')
    parser.add_argument('--num-cand-pgms', '-n', type=int, default=20,
                        help='Number of candidate programs (default: 20)')
    parser.add_argument('--num-select-pgms', '-s', type=int, default=None,
                        help='Number of programs LLM should select (default: config value, typically 2)')
    parser.add_argument('--show-all', action='store_true',
                        help='Show full ranking of all programs')
    parser.add_argument('--data-source', type=str, default='local',
                        choices=['local', 'db'],
                        help='Data source (default: local)')
    args = parser.parse_args()

    message = args.message or DEFAULT_MESSAGE

    trace_program_classification(
        message=message,
        num_cand_pgms=args.num_cand_pgms,
        num_select_pgms=args.num_select_pgms,
        show_all=args.show_all,
        data_source=args.data_source,
    )


if __name__ == '__main__':
    main()
