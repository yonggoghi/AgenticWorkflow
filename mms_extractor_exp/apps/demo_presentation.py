#!/usr/bin/env python3
"""
MMS Extractor Presentation Demo

Two-page Streamlit app for offline demo using pre-computed results.
- Page 1: 파이프라인 설명 (pipeline overview + demo results merged)
- Page 2: 라이브 데모 (optional, requires API server)

Usage:
    streamlit run apps/demo_presentation.py --server.port 8502
"""

import streamlit as st
import json
import os
import sys
import html as html_lib
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="MMS Extractor - 프레젠테이션",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    /* Force dark mode */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stSidebar"], [data-testid="stSidebarContent"],
    .main, .block-container, section[data-testid="stSidebar"] {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #1a1c23 !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown, .stText {
        color: #fafafa !important;
    }
    .main-header h1, .main-header p { color: white !important; }
    .main-header p { color: #e0e7ff !important; }
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 { color: white; margin: 0; }
    .main-header p { color: #e0e7ff; margin: 0.5rem 0 0 0; font-size: 1.3rem; }
    .message-card {
        background: #1e293b;
        color: #e2e8f0;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #4f46e5;
        margin: 0.5rem 0;
        font-size: 1.15rem;
        line-height: 1.8;
        max-height: 300px;
        overflow-y: auto;
    }
    .pipeline-flow {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: nowrap;
        gap: 0;
        margin: 0.5rem 0;
    }
    .pipeline-step-box {
        padding: 0.5rem 0.6rem;
        border-radius: 6px;
        text-align: center;
        font-size: 0.78rem;
        min-width: 90px;
        cursor: default;
        line-height: 1.3;
    }
    .pipeline-arrow {
        font-size: 1.2rem;
        color: #94a3b8;
        padding: 0 0.2rem;
    }
    .pipeline-connector {
        text-align: center;
        font-size: 1.3rem;
        color: #94a3b8;
        margin: 0.1rem 0;
    }
    /* Bigger step buttons */
    button[kind="secondary"], button[kind="primary"] {
        padding-top: 1.2rem !important;
        padding-bottom: 1.2rem !important;
        min-height: 4.5rem !important;
    }
    button[kind="secondary"] p, button[kind="primary"] p {
        font-size: 1.3rem !important;
        font-weight: 500 !important;
    }
    /* Unclicked: blue */
    button[kind="secondary"] {
        background-color: #4f46e5 !important;
        color: white !important;
        border-color: #4f46e5 !important;
    }
    button[kind="secondary"] p {
        color: white !important;
    }
    button[kind="secondary"]:hover {
        background-color: #4338ca !important;
        border-color: #4338ca !important;
    }
    /* Clicked: green */
    button[kind="primary"] {
        background-color: #059669 !important;
        color: white !important;
        border-color: #059669 !important;
    }
    button[kind="primary"] p {
        color: white !important;
    }
    button[kind="primary"]:hover {
        background-color: #047857 !important;
        border-color: #047857 !important;
    }
</style>
""", unsafe_allow_html=True)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DEMO_DATA_DIR = PROJECT_ROOT / "data" / "demo_results"
DAG_IMAGES_DIR = PROJECT_ROOT / "dag_images"

# Pipeline step definitions
PIPELINE_STEPS = [
    {
        "num": 1, "name": "InputValidationStep", "kr": "메시지 전처리",
        "desc": "메시지 길이 체크, 정제",
        "tech": ["텍스트 strip/정제", "길이 검증 (10~5000자)", "타입 검증"],
        "input_desc": "원본 MMS 메시지 텍스트",
        "output_desc": "정제된 메시지 텍스트 (whitespace, 특수 문자 제거)",
    },
    {
        "num": 2, "name": "EntityExtractionStep", "kr": "엔티티 추출",
        "desc": "Kiwi NLP로 상품/브랜드 후보 추출",
        "tech": ["Kiwi 형태소 분석기", "Bigram 사전필터링 (45K aliases → 후보 축소)", "Fuzzy String Matching (fuzz.ratio)", "SequenceMatcher 유사도"],
        "input_desc": "정제된 메시지 + 45K 상품 별칭 DB",
        "output_desc": "NLP 및 ML 추출 후보 상품 리스트",
    },
    {
        "num": 3, "name": "ProgramClassificationStep", "kr": "프로그램 분류",
        "desc": "임베딩 유사도 기반 프로그램 매칭",
        "tech": ["ko-sroberta-multitask 임베딩 모델", "Cosine Similarity 유사도 계산", "Top-K 후보 선정"],
        "input_desc": "정제된 메시지",
        "output_desc": "프로그램 후보 리스트 (유사도 점수)",
    },
    {
        "num": 4, "name": "ContextPreparationStep", "kr": "컨텍스트 준비",
        "desc": "LLM 프롬프트용 컨텍스트 구성",
        "tech": ["프롬프트 템플릿 조립", "엔티티/프로그램 컨텍스트 포맷팅", "DAG 컨텍스트 모드 적용"],
        "input_desc": "엔티티 + 프로그램 후보",
        "output_desc": "LLM 프롬프트 컨텍스트 문자열",
    },
    {
        "num": 5, "name": "LLMExtractionStep", "kr": "LLM 추출",
        "desc": "A.X LLM으로 구조화된 정보 추출",
        "tech": ["A.X (SKT) LLM API 호출", "구조화된 프롬프트 (JSON 출력 지시)", "Temperature 0.0", "Fallback 처리"],
        "input_desc": "구성된 LLM 프롬프트",
        "output_desc": "LLM JSON 텍스트 응답",
    },
    {
        "num": 6, "name": "ResponseParsingStep", "kr": "응답 분석",
        "desc": "LLM JSON 응답 파싱 및 검증",
        "tech": ["JSON 파싱 (다중 객체 지원)", "스키마 검증", "스키마 응답 감지/거부"],
        "input_desc": "LLM JSON 텍스트 응답",
        "output_desc": "추출된 상품명 + 원본 JSON",
    },
    {
        "num": 7, "name": "EntityContextExtractionStep", "kr": "엔티티+컨텍스트 추출",
        "desc": "LLM으로 1차 엔티티 및 컨텍스트 추출 (Stage 1)",
        "tech": ["LLM 기반 엔티티 추출", "관계 정보 추출", "엔티티 타입 분류", "조건부 실행 (llm 모드만)"],
        "input_desc": "메시지 텍스트 + LLM",
        "output_desc": "1차 추출 엔티티명 + 컨텍스트 정보",
    },
    {
        "num": 8, "name": "VocabularyFilteringStep", "kr": "어휘 필터링",
        "desc": "상품 어휘 DB와 매칭하여 필터링 (Stage 2)",
        "tech": ["Bigram 사전필터링", "Fuzzy Matching (fuzz.ratio)", "item_id DB 매칭", "조건부 실행 (llm 모드만)"],
        "input_desc": "Stage 1 엔티티 + 45K 상품 DB",
        "output_desc": "매칭된 상품 (item_nm, item_id, 유사도)",
    },
    {
        "num": 9, "name": "ResultConstructionStep", "kr": "결과 구성",
        "desc": "최종 결과 JSON 조립",
        "tech": ["결과 필드 조립", "상품/채널/프로그램 통합", "메타데이터 첨부"],
        "input_desc": "매칭된 엔티티 + 메타데이터",
        "output_desc": "최종 추출 결과 JSON",
    },
    {
        "num": 10, "name": "ValidationStep", "kr": "결과 검증",
        "desc": "필수 필드 확인 및 품질 체크",
        "tech": ["필수 키 검증 (title, product, channel)", "빈 결과 감지", "Fallback 트리거"],
        "input_desc": "ext_result JSON",
        "output_desc": "검증된 최종 결과",
    },
    {
        "num": 11, "name": "DAGExtractionStep", "kr": "DAG 추출",
        "desc": "엔티티 관계 그래프 생성",
        "tech": ["LLM 기반 관계 추출 (CoT 프롬프트)", "NetworkX 방향 그래프", "Graphviz 시각화", "조건부 실행"],
        "input_desc": "메시지 텍스트 + LLM",
        "output_desc": "엔티티 관계 DAG + 시각화 이미지",
    },
]


def format_mms_message(msg: str) -> str:
    """Format MMS message for display: __ → newline, _ → space."""
    msg = html_lib.escape(msg)
    msg = msg.replace("__", "<br>")
    msg = msg.replace("_", " ")
    return msg


@st.cache_data
def load_demo_results() -> List[Dict[str, Any]]:
    """Load all pre-computed demo JSON files."""
    results = []
    if not DEMO_DATA_DIR.exists():
        return results
    for jf in sorted(DEMO_DATA_DIR.glob("*.json")):
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['_filename'] = jf.name
            results.append(data)
        except Exception as e:
            st.warning(f"Failed to load {jf.name}: {e}")
    return results


# ── Pipeline Visualization ──────────────────────────────────────────────

def _build_pipeline_html(step_timings_map: dict, selected_step: int = None) -> str:
    """Build HTML for the pipeline flow diagram with arrows."""
    colors_row1 = "#4f46e5"
    colors_row2 = "#7c3aed"
    color_last = "#059669"

    def step_box(idx):
        s = PIPELINE_STEPS[idx]
        timing = step_timings_map.get(s["name"], {})
        duration = timing.get("duration", 0)
        status = timing.get("status", "")
        icon = {"success": "✅", "skipped": "⏭️", "failed": "❌"}.get(status, "")

        if idx == 10:
            bg = color_last
        elif idx >= 6:
            bg = colors_row2
        else:
            bg = colors_row1

        border = "3px solid #facc15" if selected_step == idx else "2px solid transparent"

        return (
            f'<div class="pipeline-step-box" style="background:{bg}; color:white; border:{border};">'
            f'<div style="font-weight:bold;">Step {s["num"]}</div>'
            f'<div>{s["kr"]}</div>'
            f'<div style="font-size:0.7rem; color:#e0e7ff;">{duration:.1f}s {icon}</div>'
            f'</div>'
        )

    arrow = '<span class="pipeline-arrow">→</span>'

    row1_items = []
    for i in range(6):
        row1_items.append(step_box(i))
        if i < 5:
            row1_items.append(arrow)

    row2_items = []
    for i in range(6, 11):
        row2_items.append(step_box(i))
        if i < 10:
            row2_items.append(arrow)

    html = f"""
    <div class="pipeline-flow">{''.join(row1_items)}</div>
    <div class="pipeline-connector">↓</div>
    <div class="pipeline-flow">{''.join(row2_items)}</div>
    """
    return html


def _render_step_buttons(demo: dict):
    """Render clickable step buttons in two rows."""
    step_timings_map = {s["step"]: s for s in demo.get("step_timings", [])}

    def render_row(start, end):
        # Columns: [step, arrow, step, arrow, ..., step]
        widths = []
        for i in range(start, end):
            widths.append(4)
            if i < end - 1:
                widths.append(1)
        cols = st.columns(widths)

        col_i = 0
        for idx in range(start, end):
            s = PIPELINE_STEPS[idx]
            timing = step_timings_map.get(s["name"], {})
            status = timing.get("status", "")
            icon = {"success": "✅", "skipped": "⏭️", "failed": "❌"}.get(status, "")
            is_selected = st.session_state.get("selected_step") == idx

            with cols[col_i]:
                label = f"Step {s['num']}: {s['kr']} {icon}"
                if st.button(label, key=f"step_btn_{idx}",
                             type="primary" if is_selected else "secondary",
                             use_container_width=True):
                    st.session_state.selected_step = idx
                    st.rerun()
            col_i += 1

            if idx < end - 1:
                with cols[col_i]:
                    st.markdown(
                        "<div style='text-align:center; font-size:1.6rem; color:#94a3b8; display:flex; align-items:center; justify-content:center; min-height:4.5rem;'>→</div>",
                        unsafe_allow_html=True
                    )
                col_i += 1

    render_row(0, 6)
    render_row(6, 11)


# ── Step Detail ─────────────────────────────────────────────────────────

def _render_step_detail(demo: dict, step_idx: int):
    """Show input/output and technology for the selected step."""
    s = PIPELINE_STEPS[step_idx]
    step_timings_map = {st_info["step"]: st_info for st_info in demo.get("step_timings", [])}
    timing = step_timings_map.get(s["name"], {})

    st.subheader(f"Step {s['num']}: {s['kr']} ({s['name']})")
    st.caption(f"⏱️ {timing.get('duration', 0):.3f}초  |  상태: {timing.get('status', 'N/A')}  |  {s['desc']}")

    col_io, col_tech = st.columns([3, 2])

    with col_io:
        st.markdown(f'### <span style="color:#059669;">▶ 입력:</span> {s["input_desc"]}', unsafe_allow_html=True)
        st.markdown(f'### <span style="color:#059669;">◀ 출력:</span> {s["output_desc"]}', unsafe_allow_html=True)
        _show_step_actual_data(demo, step_idx)

    with col_tech:
        # st.markdown(  "#### 🔧 사용 기술")
        st.markdown(f'### <span style="color:#059669;">🔧 사용 기술</span>', unsafe_allow_html=True)
        for tech in s["tech"]:
            st.markdown(f"- {tech}")


def _show_step_actual_data(demo: dict, step_idx: int):
    """Show actual data from the demo result for the selected step."""
    ext = demo.get("ext_result", {})
    raw = demo.get("raw_result", {})
    msg = demo.get("message", "")
    step_num = step_idx + 1

    if step_num == 1:
        pass

    elif step_num == 2:
        kiwi_entities = demo.get("entities_from_kiwi", [])
        cand_items = demo.get("cand_item_list", [])
        # if kiwi_entities:
        st.markdown('<h4 style="color:#4f46e5; margin-left:1.5rem;">NLP 추출 엔티티</h4>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin-left:2rem;">{", ".join(str(e) for e in kiwi_entities)}</div>', unsafe_allow_html=True)
        # if cand_items:
        st.markdown('<h4 style="color:#7c3aed; margin-left:1.5rem;">Fuzzy Matching 후보 엔티티</h4>', unsafe_allow_html=True)
        if isinstance(cand_items[0], dict):
            st.dataframe(pd.DataFrame(cand_items), use_container_width=True, hide_index=True)
        else:
            st.markdown(f'<div style="margin-left:2rem;">{", ".join(str(c) for c in cand_items)}</div>', unsafe_allow_html=True)

    elif step_num == 3:
        pgm = ext.get("pgm", [])
        if pgm:
            st.markdown('<h4 style="color:#7c3aed; margin-left:1.5rem;">프로그램 매칭 결과</h4>', unsafe_allow_html=True)
            if isinstance(pgm, list) and all(isinstance(p, dict) for p in pgm):
                st.dataframe(pd.DataFrame(pgm), use_container_width=True, hide_index=True)
            else:
                for p in pgm:
                    st.write(f"- {p}")

    elif step_num == 4:
        rag_context = demo.get("rag_context", "")
        if rag_context:
            truncated = rag_context[:1000]
            if len(rag_context) > 1000:
                truncated += f"\n\n... (총 {len(rag_context):,}자 중 1,000자 표시)"
            st.code(truncated, language=None)

    elif step_num == 5:
        st.markdown('<h4 style="color:#7c3aed; margin-left:1.5rem;">LLM 원본 출력 (raw_result)</h4>', unsafe_allow_html=True)
        if raw:
            st.json(raw)

    elif step_num == 6:
        raw_products = raw.get("product", [])
        if raw_products:
            names = [p.get("name", str(p)) if isinstance(p, dict) else str(p) for p in raw_products]
            st.markdown(f'<div style="margin-left:2rem;">{", ".join(names)}</div>', unsafe_allow_html=True)

    elif step_num == 7:
        # EntityContextExtractionStep (Stage 1 entities)
        extracted_entities = demo.get("extracted_entities", {})
        if extracted_entities:
            st.markdown('<h4 style="color:#7c3aed; margin-left:1.5rem;">Stage 1 추출 엔티티</h4>', unsafe_allow_html=True)
            entities_list = extracted_entities.get("entities", [])
            if entities_list:
                st.markdown(f'<div style="margin-left:2rem;">{", ".join(str(e) for e in entities_list)}</div>', unsafe_allow_html=True)
            context = extracted_entities.get("context_text", "")
            if context:
                st.markdown('<h4 style="color:#4f46e5; margin-left:1.5rem;">추출 컨텍스트</h4>', unsafe_allow_html=True)
                st.text(context[:500] + ("..." if len(context) > 500 else ""))

    elif step_num == 8:
        # VocabularyFilteringStep (matched products)
        products = ext.get("product", [])
        if products:
            st.markdown('<h4 style="color:#7c3aed; margin-left:1.5rem;">매칭된 상품</h4>', unsafe_allow_html=True)
            if isinstance(products, list) and all(isinstance(p, dict) for p in products):
                rows = []
                for p in products:
                    row = {}
                    for k, v in p.items():
                        row[k] = ', '.join(str(x) for x in v) if isinstance(v, list) else v
                    rows.append(row)
                df = pd.DataFrame(rows)
                preferred = ['item_name_in_msg', 'expected_action', 'item_in_voca']
                available = [c for c in preferred if c in df.columns]
                remaining = [c for c in df.columns if c not in preferred]
                df = df[available + remaining]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                for p in products:
                    st.write(f"- {p}")

    elif step_num == 9:
        # ResultConstructionStep (final result JSON)
        # st.markdown('<h4 style="color:#7c3aed; margin-left:1.5rem;">최종 추출 결과</h4>', unsafe_allow_html=True)
        st.json(ext)

    elif step_num == 11:
        entity_dag = ext.get("entity_dag", [])
        if entity_dag:
            st.markdown('<h4 style="color:#7c3aed; margin-left:1.5rem;">DAG 텍스트</h4>', unsafe_allow_html=True)
            for line in entity_dag:
                st.write(f"- {line}")
        st.markdown('<h4 style="color:#4f46e5; margin-left:1.5rem;">DAG 이미지</h4>', unsafe_allow_html=True)
        dag_filename = demo.get("dag_image_filename")
        if dag_filename:
            dag_path = DAG_IMAGES_DIR / dag_filename
            if dag_path.exists():
                st.image(str(dag_path), caption=f"DAG ({dag_filename})", use_container_width=True)


# ── Extracted Results ───────────────────────────────────────────────────

def _display_extracted_info(ext_result: Dict[str, Any]):
    """Display extracted information in a structured layout."""
    if not ext_result:
        st.info("추출 결과가 없습니다.")
        return

    category_config = {
        'title': ('📝', 'Title'),
        'purpose': ('🎯', 'Purpose'),
        'product': ('📦', 'Product'),
        'channel': ('📱', 'Channel'),
        'pgm': ('⚙️', 'Program'),
        'entity_dag': ('🔗', 'Entity DAG'),
    }
    display_order = ['title', 'purpose', 'product', 'channel', 'pgm', 'entity_dag']

    for key in display_order:
        if key not in ext_result or not ext_result[key]:
            continue
        items = ext_result[key]
        icon, label = category_config.get(key, ('📊', key.upper()))
        st.markdown(f"### {icon} {label}")

        if isinstance(items, list) and len(items) > 0:
            if all(isinstance(item, dict) for item in items):
                flattened = []
                for item in items:
                    row = {}
                    for k, v in item.items():
                        row[k] = ', '.join(str(x) for x in v) if isinstance(v, list) else v
                    flattened.append(row)
                df = pd.DataFrame(flattened)
                if key == 'product':
                    preferred = ['item_name_in_msg', 'expected_action', 'item_in_voca']
                    available = [c for c in preferred if c in df.columns]
                    remaining = [c for c in df.columns if c not in preferred]
                    df = df[available + remaining]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                for item in items:
                    st.write(f"- {item}")
        elif isinstance(items, str):
            st.write(items)
        elif isinstance(items, dict):
            st.json(items)


def _display_dag_image(demo: Dict[str, Any]):
    """Display the DAG image for this demo."""
    dag_filename = demo.get("dag_image_filename")
    if not dag_filename:
        st.info("이 메시지에 대한 DAG 이미지가 생성되지 않았습니다.")
        return

    dag_path = DAG_IMAGES_DIR / dag_filename
    if dag_path.exists():
        st.image(str(dag_path), caption=f"오퍼 관계 DAG ({dag_filename})", use_container_width=True)
        st.caption(f"파일: `{dag_filename}` ({dag_path.stat().st_size:,} bytes)")
    else:
        st.warning(f"DAG 이미지 파일을 찾을 수 없습니다: `{dag_filename}`")

    entity_dag = demo.get("ext_result", {}).get("entity_dag", [])
    if entity_dag:
        with st.expander("DAG 텍스트 데이터"):
            for line in entity_dag:
                st.write(f"- {line}")


# ── Page: 파이프라인 설명 ──────────────────────────────────────────────

def page_pipeline(demos: List[Dict[str, Any]]):
    """Main pipeline page (merged overview + demo results)."""
    st.markdown("""
    <div class="main-header">
        <h1>📊 MMS Extractor 작업 흐름 설명</h1>
        <p>MMS 광고 메시지에서 구조화된 정보를 추출하는 11단계 AI 파이프라인</p>
    </div>
    """, unsafe_allow_html=True)

    if not demos:
        st.error("데모 데이터가 없습니다. `scripts/generate_demo_data.py`를 먼저 실행해주세요.")
        return

    # Sidebar: message selection
    with st.sidebar:
        st.header("데모 메시지 선택")
        titles = [d.get("title", f"Message {i+1}") for i, d in enumerate(demos)]
        selected_idx = st.radio(
            "메시지 선택",
            range(len(demos)),
            format_func=lambda i: f"{i+1}. {titles[i]}",
            key="demo_select"
        )

    demo = demos[selected_idx]

    # ── 1. Original Message ──
    st.subheader(f"📝 원본 메시지: {demo.get('title', '')}")
    formatted = format_mms_message(demo.get("message", ""))
    st.markdown(f'<div class="message-card">{formatted}</div>', unsafe_allow_html=True)

    st.divider()

    # ── 2. Pipeline Diagram ──
    st.subheader("11-Step Workflow Pipeline")

    # Initialize session state
    if "selected_step" not in st.session_state:
        st.session_state.selected_step = None

    # # Visual diagram (HTML with arrows)
    # step_timings_map = {s["step"]: s for s in demo.get("step_timings", [])}
    # pipeline_html = _build_pipeline_html(step_timings_map, st.session_state.selected_step)
    # st.markdown(pipeline_html, unsafe_allow_html=True)

    st.caption("단계를 클릭하면 상세 정보를 확인할 수 있습니다")

    # Interactive buttons
    _render_step_buttons(demo)

    st.divider()

    # ── 3. Step Detail (when step selected) ──
    if st.session_state.selected_step is not None:
        _render_step_detail(demo, st.session_state.selected_step)
        st.divider()

    # ── 4. Extracted Results ──
    st.subheader("📊 추출 결과")
    ext_result = demo.get("ext_result", {})

    tab_result, tab_dag, tab_json = st.tabs(["추출 정보", "DAG 이미지", "전체 JSON"])

    with tab_result:
        _display_extracted_info(ext_result)

    with tab_dag:
        _display_dag_image(demo)

    with tab_json:
        st.json(demo)


# ── Page: 라이브 데모 ──────────────────────────────────────────────────

def page_live_demo():
    """Live Demo page (requires API server)."""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 라이브 데모</h1>
        <p>실시간으로 MMS 메시지를 분석합니다 (API 서버 필요)</p>
    </div>
    """, unsafe_allow_html=True)

    api_available = False
    api_url = "http://localhost:8000"
    try:
        import requests
        response = requests.get(f"{api_url}/health", timeout=3)
        api_available = response.status_code == 200
    except Exception:
        pass

    if api_available:
        st.success(f"API 서버 연결됨: {api_url}")
    else:
        st.error("API 서버가 실행되고 있지 않습니다.")
        st.markdown("""
        **Page 1 (파이프라인 설명)** 에서 사전 처리된 결과를 확인하세요.
        """)

    st.divider()

    message = st.text_area("MMS 메시지 입력", height=200, placeholder="분석할 MMS 메시지를 입력하세요...")

    run_button = st.button("분석 실행", type="primary", disabled=not api_available or not message.strip())

    if run_button and message.strip() and api_available:
        import requests
        with st.spinner("메시지 분석 중... (최대 2분 소요)"):
            try:
                response = requests.post(
                    f"{api_url}/extract",
                    json={"message": message, "llm_model": "ax", "offer_info_data_src": "local", "extract_entity_dag": False},
                    timeout=120
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success("분석 완료!")
                    ext_result = result.get("result", result.get("ext_result", {}))
                    if ext_result:
                        _display_extracted_info(ext_result)
                    with st.expander("전체 JSON 응답"):
                        st.json(result)
                else:
                    st.error(f"API 오류: {response.status_code}")
                    st.code(response.text)
            except Exception as e:
                st.error(f"요청 실패: {e}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    demos = load_demo_results()

    with st.sidebar:
        st.markdown("## 📊 MMS Extractor")
        st.markdown("**프레젠테이션 데모**")
        st.divider()

        page = st.radio(
            "페이지 선택",
            ["파이프라인 설명", "라이브 데모"],
            index=0,
            key="page_nav"
        )

        st.divider()
        if demos:
            st.success(f"데모 데이터: {len(demos)}건 로드됨")
        else:
            st.warning("데모 데이터 없음")
            st.caption("generate_demo_data.py 실행 필요")

    if page == "파이프라인 설명":
        page_pipeline(demos)
    elif page == "라이브 데모":
        page_live_demo()


if __name__ == "__main__":
    main()
