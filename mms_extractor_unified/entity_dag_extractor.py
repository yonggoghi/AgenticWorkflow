from concurrent.futures import ThreadPoolExecutor
import time
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
import re
# from pygments import highlight
# from pygments.lexers import JsonLexer
# from pygments.formatters import HtmlFormatter
# from IPython.display import HTML
import pandas as pd
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI
from typing import List, Tuple, Union, Dict, Any
import ast
from rapidfuzz import fuzz, process
import re
import json
import glob
import os
from config import settings

pd.set_option('display.max_colwidth', 500)

llm_api_key = settings.API_CONFIG.llm_api_key
llm_api_url = settings.API_CONFIG.llm_api_url
client = OpenAI(
    api_key = llm_api_key,
    base_url = llm_api_url
)
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import AIMessage, HumanMessage, SystemMessage

llm_gem = ChatOpenAI(
        temperature=0,
        openai_api_key=llm_api_key,
        openai_api_base=llm_api_url,
        model=settings.ModelConfig.gemma_model,
        max_tokens=settings.ModelConfig.llm_max_tokens
        )

llm_ax = ChatOpenAI(
        temperature=0,
        openai_api_key=llm_api_key,
        openai_api_base=llm_api_url,
        model=settings.ModelConfig.ax_model,
        max_tokens=settings.ModelConfig.llm_max_tokens
        )

llm_cld = ChatOpenAI(
        temperature=0,
        openai_api_key=llm_api_key,
        openai_api_base=llm_api_url,
        model=settings.ModelConfig.claude_model,
        max_tokens=settings.ModelConfig.llm_max_tokens
        )

llm_gen = ChatOpenAI(
        temperature=0,
        openai_api_key=llm_api_key,
        openai_api_base=llm_api_url,
        model=settings.ModelConfig.gemini_model,
        max_tokens=settings.ModelConfig.llm_max_tokens
        )

llm_gpt = ChatOpenAI(
        temperature=0,
        openai_api_key=llm_api_key,
        openai_api_base=llm_api_url,
        model=settings.ModelConfig.gpt_model,
        max_tokens=settings.ModelConfig.llm_max_tokens
        )

stop_item_names = pd.read_csv(settings.METADATA_CONFIG.stop_items_path)['stop_words'].to_list()
mms_pdf = pd.read_csv(settings.METADATA_CONFIG.mms_msg_path)
mms_pdf = mms_pdf.astype('str')

import re, networkx as nx
from typing import List, Tuple, Set, Dict

###############################################################################
# 1) 정규식 : ( node ) -[ relation ]-> ( node )
###############################################################################
PAT = re.compile(r"\((.*?)\)\s*-\[(.*?)\]->\s*\((.*?)\)")
NODE_ONLY = re.compile(r"\((.*?)\)\s*$")

###############################################################################
# 2) 파서
###############################################################################

def parse_block(text: str):
    nodes: Set[str] = set()
    edges: List[Tuple[str,str,str]] = []
    for line in filter(None, map(str.strip, text.splitlines())):
        m = PAT.match(line)
        if m:                            # (A) -[rel]-> (B)
            src, rel, dst = map(str.strip, m.groups())
            nodes.update([src, dst])
            edges.append((src, dst, rel))
            continue

        m2 = NODE_ONLY.match(line)       # (A:B:C)
        if m2:                           # ← 독립 노드
            nodes.add(m2.group(1).strip())
            continue

        raise ValueError(f"못읽은 라인:\n  {line}")
    return list(nodes), edges

###############################################################################
# 3) 노드 스플리터 – 3칸·2칸·1칸 허용
###############################################################################
def split_node(raw: str) -> Dict[str,str]:
    parts = [p.strip() for p in raw.split(":")]
    
    if len(parts) == 1:                       # entity only
        ent, act, kpi = parts[0], "", ""
    elif len(parts) == 2:                     # entity:metric
        ent, act = parts
        kpi = ""
    elif len(parts) == 3:                     # entity:action:metric
        ent, act, kpi = parts
    elif len(parts) == 4:                     # entity:action:behavior:metric
        ent, act, behavior, kpi = parts
        # Combine action and behavior, or handle separately
        act = f"{act}:{behavior}"  # Option 1: combine them
        # Or you could add a new field:
        # return {"entity": ent, "action": act, "behavior": behavior, "metric": kpi}
    elif len(parts) >= 5:                     # Handle even more parts flexibly
        ent = parts[0]
        kpi = parts[-1]  # Last part is usually the metric
        act = ":".join(parts[1:-1])  # Everything in between becomes action
    else:
        # This shouldn't happen with len(parts) >= 1, but just in case
        raise ValueError(f"알 수 없는 노드 형식: {raw}")
    
    return {"entity": ent, "action": act, "metric": kpi}


###############################################################################
# 4) DAG 빌더
###############################################################################
def build_dag(nodes: List[str], edges: List[Tuple[str,str,str]]) -> nx.DiGraph:
    g = nx.DiGraph()
    
    # Add nodes with error handling
    for n in nodes:
        try:
            node_attrs = split_node(n)
            g.add_node(n, **node_attrs)
        except ValueError as e:
            print(f"Warning: {e}")
            # Add node with minimal attributes if parsing fails
            g.add_node(n, entity=n, action="", metric="")
    
    # Add edges
    for src, dst, rel in edges:
        g.add_edge(src, dst, relation=rel)
    
    # Optional: Check if it's a DAG
    # if not nx.is_directed_acyclic_graph(g):
    #     raise nx.NetworkXUnfeasible("사이클이 있습니다 ― DAG 아님!")
    
    return g


###############################################################################
# 5) Path Finder
###############################################################################
def get_root_to_leaf_paths(dag):
    """Generate all paths from root nodes (no predecessors) to leaf nodes (no successors)"""
    # Find root nodes (no incoming edges)
    root_nodes = [node for node in dag.nodes() if dag.in_degree(node) == 0]
    
    # Find leaf nodes (no outgoing edges)
    leaf_nodes = [node for node in dag.nodes() if dag.out_degree(node) == 0]
    
    all_paths = []
    for root in root_nodes:
        for leaf in leaf_nodes:
            try:
                paths = list(nx.all_simple_paths(dag, root, leaf))
                all_paths.extend(paths)
            except nx.NetworkXNoPath:
                continue
    
    return all_paths, root_nodes, leaf_nodes

import random

def extract_dag(num_msgs=50, llm_model_nm='ax'):

    if llm_model_nm == 'ax':
        llm_model = llm_ax
    elif llm_model_nm == 'gem':
        llm_model = llm_gem
    elif llm_model_nm == 'cld':
        llm_model = llm_cld
    elif llm_model_nm == 'gen':
        llm_model = llm_gen
    elif llm_model_nm == 'gpt':
        llm_model = llm_gpt

    # 출력을 파일에 저장하기 위한 설정
    output_file = "/Users/1110566/workspace/AgenticWorkflow/mms_extractor_unified/dag_extraction_output.txt"

    line_break_patterns = {"__":"\n", "■":"\n■", "▶":"\n▶", "_":"\n"}
    
    with open(output_file, 'a', encoding='utf-8') as f:
        # 실행 시작 시점 기록
        from datetime import datetime
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'='*80}\n")
        f.write(f"DAG 추출 실행 시작: {start_time}\n")
        f.write(f"{'='*80}\n\n")
        
        for msg in random.sample(mms_pdf.query("msg.str.contains('')")['msg'].unique().tolist(), num_msgs):
            try:
    #             msg = """
    # [SK텔레콤] 강남터미널대리점 본점 갤럭시 S25 사전예약 안내드립니다.
    # (광고)[SKT] 강남터미널대리점 본점 갤럭시 S25 사전예약 안내__고객님, 안녕하세요. _새로운 시작, 설레이는 1월! SK텔레콤 강남터미널 대리점이 고객님의 특별한 새해를 응원합니다._곧 출시하는 삼성의 최신 플래그십 스마트폰 갤럭시 S25 사전예약 혜택 받아 가세요.__■ 새 학기 맞이 키즈폰 특별 행사_- 월정액 요금 및 기기 할인 최대 설계_- 12개월 약정__■ 갤럭시 S25 사전예약 중!_- 개통일 : 2월4일_- 더블 스토리지, 워치7 등 푸짐한 사은 혜택은 아래 매장 연락처로 문의주세요._- 예약 선물도 챙기시고, 좋은 조건으로 구매 상담도 받아 보세요.__■ 갤럭시 S24 마지막 찬스_- 요금 및 기기 할인 최대 설계_- 워치7 무료 증정 (※프라임 요금제 사용 기준)__■ 인터넷+TV결합 혜택_- 60만 원 상당의 최대 사은품 증정_- 월 최저 요금 설계__■ 강남터미널대리점 본점_- 주소 : 서울시 서초구 신반포로 176, 1층 130호 (신세계백화점 옆, 센트럴시티내 호남선 하차장 아웃백 아래 1층)_- 연락처 : 02-6282-1011_▶ 매장 홈페이지/예약/상담 : http://t-mms.kr/t.do?m=#61&s=30251&a=&u=http://tworldfriends.co.kr/D145410000__■ 문의: SKT 고객센터(1558, 무료)_SKT와 함께 해주셔서 감사합니다.__무료 수신거부 1504
    #             """

                for pattern, replacement in line_break_patterns.items():
                    msg = msg.replace(pattern, replacement)

                prompt_1 = f"""
## 작업
통신사 광고 메시지에서 개체명과 기대 행동을 추출하고 DAG 형식으로 출력하세요.

## 출력 형식
- **독립 노드**: `(개체명:기대행동)`
- **관계 노드**: `(개체명:기대행동) -[관계동사]-> (개체명:기대행동)`

## 개체명 유형 및 예시
### 📱 제품/단말기
- 스마트폰: 갤럭시S24, 아이폰15, 갤럭시폴더블6, 갤럭시워치, 갤럭시버즈
- 기타 기기: ZEM꾸러미폰, 키즈폰, 실버폰, 태블릿

### 📞 서비스/요금제
- 요금제: 5G요금제, 프라임요금제, T플랜, ZEM요금제
- 통신서비스: 인터넷, IPTV, ADT캡스, 우주패스, 에이닷
- 부가서비스: V컬러링, 콜키퍼, 통화가능통보플러스

### 🎁 혜택/할인
- 할인: 50%할인, 10만원할인, 요금할인, 기기값할인
- 혜택: 사은품, 쿠폰, 포인트적립, 무료체험
- 구체적 혜택: 갤럭시워치증정, 에어팟증정, 충전기세트

### 🏢 장소/매장
- 온라인: T다이렉트샵, 온라인몰, 홈페이지, 앱
- 오프라인: SKT대리점, T월드매장, 구체적매장명(예: 강남점)

### 🎉 이벤트/프로모션
- 기간 이벤트: 봄맞이행사, 신규가입이벤트, 사전예약이벤트
- 멤버십: T멤버십, 단골등록, 친구추가

### 주의 사항
- 광고 타겟은 개체명으로 추출하지 마세요.
- 일정/기간은 개체명으로 추출하지 마세요.

## 기대 행동 (표준화된 10개 동사)
**[구매, 가입, 사용, 방문, 참여, 등록, 다운로드, 확인, 수령, 적립]**

## 관계 동사 가이드라인

### 🔥 조건부 관계 (최우선 사용)
**조건 충족 시 혜택 제공을 명확히 표현**
- `가입하면`, `구매하면`, `방문하면`, `신청하면`, `등록하면`
- `가입시`, `구매시`, `등록시`, `사용시`, `방문시`
- `가입후`, `구매후`, `완료후`, `설치후`

### 💎 혜택 제공 관계
**혜택/보상 수령을 표현**
- `증정받다`, `할인받다`, `제공받다`, `지원받다`
- `수령하다`, `적립하다`, `받다`

### 🔗 연결/경로 관계
**서비스 간 연결이나 경로를 표현**
- `통해`, `통하여`, `이용하여`, `활용하여`
- `함께`, `결합하여`, `연결하여`, `동시가입`

### ⚡ 행동 유도 관계
**특정 행동을 유도하는 관계**
- `참여하여`, `체험하여`, `신청하여`
- `문의하여`, `확인하여`, `안내받아`

### 📱 플랫폼/채널 관계
**특정 플랫폼이나 채널을 통한 접근**
- `접속하여`, `다운로드하여`, `설치하여`
- `로그인하여`, `인증하여`

## 고급 추출 규칙

### ✅ 반드시 포함해야 할 요소
1. **Root Node**: 사용자가 시작할 수 있는 행동 (방문, 가입, 다운로드 등)
2. **조건부 혜택**: "~하면 ~받을 수 있다" 구조
3. **연쇄 혜택**: A → B → C 형태의 다단계 혜택
4. **선택적 옵션**: 여러 옵션 중 택1 상황

### ❌ 제외해야 할 요소
1. 일반적인 정보성 멘트 ("안녕하세요", "감사합니다")
2. 연락처, 주소 등 메타정보
3. 법적 고지사항 ("수신거부", "유의사항")
4. 중복되는 유사한 혜택

### 🎯 개체명 정규화 규칙
1. **구체적 명칭 사용**: "스마트폰" → "갤럭시S24"
2. **일관된 표기**: "갤럭시 S24" → "갤럭시S24" (띄어쓰기 제거)
3. **의미 단위 유지**: "5만원할인쿠폰" (분리하지 않음)
4. **브랜드명 포함**: "삼성케어플러스", "T멤버십"

### 🔄 관계 방향성 원칙
1. **시간 순서**: 먼저 일어나는 행동 → 나중 행동
2. **조건과 결과**: 조건 행동 → 결과 혜택
3. **의존성**: 전제 조건 → 수행 가능한 행동

## 출력 예시

### 단순한 조건부 혜택
```
(갤럭시S24:구매) -[구매시]-> (50%할인:수령)
(T멤버십:가입) -[가입하면]-> (매월할인:수령)
```

### 복합적 연쇄 관계
```
(SKT대리점:방문) -[방문하여]-> (상담:확인)
(상담:확인) -[완료후]-> (갤럭시S24:구매)
(갤럭시S24:구매) -[구매시]-> (갤럭시워치:수령)
```

### 다중 선택 관계
```
(우주패스:가입) -[가입시]-> (Netflix:사용)
(우주패스:가입) -[가입시]-> (Wavve:사용)
(우주패스:가입) -[가입시]-> (YouTube Premium:사용)
```

### 플랫폼 연계
```
(T월드앱:다운로드) -[다운로드하여]-> (쿠폰:수령)
(쿠폰:수령) -[사용하여]-> (30%할인:수령)
```

## 분석 시 체크리스트
■ Root Node 식별: 사용자가 시작할 수 있는 행동이 있는가?
■ 조건부 관계: "~하면", "~시" 구조가 명확한가?
■ 혜택 연쇄: 여러 단계의 혜택이 연결되어 있는가?
■ 개체명 구체성: 모호한 표현 대신 구체적 명칭을 사용했는가?
■ 관계 방향성: 시간순서와 의존성이 올바른가?
■ 중복 제거: 같은 의미의 노드가 중복되지 않았는가?

**메시지를 분석하여 위 형식으로 DAG만을 출력하세요. mermaid 형식을 사용하지 마세요.**

## message:
{msg}
"""

                # 메시지 출력
                msg_header = "==="*15+" Message "+"==="*15
                print(msg_header)
                f.write(msg_header + "\n")
                print(msg)
                f.write(msg + "\n")
                
                # DAG 출력
                dag_header = "==="*15+f" DAG ({llm_model_nm.upper()}) "+"==="*15
                print(dag_header)
                f.write(dag_header + "\n")
                dag_raw = llm_model.invoke(prompt_1).content
                print(dag_raw)
                f.write(dag_raw + "\n")

                nodes, edges = parse_block(re.sub(r'^```|```$', '', dag_raw.strip()))
                dag = build_dag(nodes, edges)

                # Root Nodes 출력
                root_header = "==="*15+" Root Nodes "+"==="*15
                print(root_header)
                f.write(root_header + "\n")
                root_nodes = [node for node in dag.nodes() if dag.in_degree(node) == 0]
                for root in root_nodes:
                    node_data = dag.nodes[root]
                    root_info = f"  {root} | {node_data}"
                    print(root_info)
                    f.write(root_info + "\n")

                # Paths 출력
                paths_header = "==="*15+" Paths "+"==="*15
                print(paths_header)
                f.write(paths_header + "\n")
                paths, roots, leaves = get_root_to_leaf_paths(dag)

                for i, path in enumerate(paths):
                    path_info = f"\nPath {i+1}:"
                    print(path_info)
                    f.write(path_info + "\n")
                    for j, node in enumerate(path):
                        if j < len(path) - 1:
                            edge_data = dag.get_edge_data(node, path[j+1])
                            relation = edge_data['relation'] if edge_data else ''
                            node_info = f"  {node}"
                            relation_info = f"    --[{relation}]-->"
                            print(node_info)
                            print(relation_info)
                            f.write(node_info + "\n")
                            f.write(relation_info + "\n")
                        else:
                            final_node = f"  {node}"
                            print(final_node)
                            f.write(final_node + "\n")

                separator = "\n" + "#"*100 + "\n"
                print(separator)
                f.write(separator)

            except Exception as e:
                print(f"Error: {e}")
                f.write(f"Error: {e}\n")
                continue
    
    print(f"출력이 파일에 저장되었습니다: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DAG 추출기')
    parser.add_argument('--num_msgs', type=int, default=50, help='추출할 메시지 수')
    parser.add_argument('--llm_model', type=str, default='ax', help='사용할 LLM 모델')
    args = parser.parse_args()
    extract_dag(num_msgs=args.num_msgs, llm_model_nm=args.llm_model)