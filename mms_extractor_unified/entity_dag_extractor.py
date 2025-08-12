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

    line_break_patterns = ["__", "■", "▶", "_"]
    
    with open(output_file, 'a', encoding='utf-8') as f:
        # 실행 시작 시점 기록
        from datetime import datetime
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'='*80}\n")
        f.write(f"DAG 추출 실행 시작: {start_time}\n")
        f.write(f"{'='*80}\n\n")
        
        for msg in random.sample(mms_pdf.query("msg.str.contains('')")['msg'].unique().tolist(), num_msgs):
#             msg = """
# [SK텔레콤] 강남터미널대리점 본점 갤럭시 S25 사전예약 안내드립니다.
# (광고)[SKT] 강남터미널대리점 본점 갤럭시 S25 사전예약 안내__고객님, 안녕하세요. _새로운 시작, 설레이는 1월! SK텔레콤 강남터미널 대리점이 고객님의 특별한 새해를 응원합니다._곧 출시하는 삼성의 최신 플래그십 스마트폰 갤럭시 S25 사전예약 혜택 받아 가세요.__■ 새 학기 맞이 키즈폰 특별 행사_- 월정액 요금 및 기기 할인 최대 설계_- 12개월 약정__■ 갤럭시 S25 사전예약 중!_- 개통일 : 2월4일_- 더블 스토리지, 워치7 등 푸짐한 사은 혜택은 아래 매장 연락처로 문의주세요._- 예약 선물도 챙기시고, 좋은 조건으로 구매 상담도 받아 보세요.__■ 갤럭시 S24 마지막 찬스_- 요금 및 기기 할인 최대 설계_- 워치7 무료 증정 (※프라임 요금제 사용 기준)__■ 인터넷+TV결합 혜택_- 60만 원 상당의 최대 사은품 증정_- 월 최저 요금 설계__■ 강남터미널대리점 본점_- 주소 : 서울시 서초구 신반포로 176, 1층 130호 (신세계백화점 옆, 센트럴시티내 호남선 하차장 아웃백 아래 1층)_- 연락처 : 02-6282-1011_▶ 매장 홈페이지/예약/상담 : http://t-mms.kr/t.do?m=#61&s=30251&a=&u=http://tworldfriends.co.kr/D145410000__■ 문의: SKT 고객센터(1558, 무료)_SKT와 함께 해주셔서 감사합니다.__무료 수신거부 1504
#             """

            for pattern in line_break_patterns:
                msg = msg.replace(pattern, "\n")

            prompt_1 = f"""
            ## 작업
            통신사 광고 메시지에서 개체명과 기대 행동을 추출하고 DAG 형식으로 출력하세요.

            ## 출력 형식
            - 독립: `(개체명:기대행동)`
            - 관계: `(개체명:기대행동) -[동사구]-> (개체명:기대행동)`

            ## 개체명 유형
            - 제품: 갤럭시 S24, 아이폰 15, 갤럭시워치
            - 서비스: 5G요금제, 인터넷, IPTV, 우주패스, FLO
            - 혜택: 50%할인, 사은품, 5만원쿠폰
            - 장소: SKT대리점, 온라인몰
            - 이벤트: 봄맞이행사, 신규가입이벤트

            ## 기대 행동
            [구매, 가입, 사용, 방문, 참여, 등록, 다운로드, 확인, 수령, 적립]

            ## 관계 동사구
            - requires: B가 A의 필수조건
            - triggers: A하면 B혜택 제공
            - bundles_with: A와 B 묶음판매
            - includes: A안에 B포함
            - enables: A가 B를 가능하게함

            ## 예시

            입력:
            “갤럭시 S24 Ultra 구매 시 5G 프라임요금제 가입 필수! 5G 프라임요금제에는 데이터 무제한이 포함되어 있으며, 인터넷 패밀리 결합을 함께 신청하면 매월 5천 원 할인 혜택이 제공됩니다. 또한 T멤버십 앱 등록 후 1만 원 할인쿠폰을 다운로드할 수 있습니다.”

            출력:
            ```
            (갤럭시 S24 Ultra:구매) -[requires]-> (5G 프라임요금제:가입)
            (5G 프라임요금제:가입) -[includes]-> (데이터 무제한:사용)
            (5G 프라임요금제:가입) -[bundles_with]-> (인터넷 패밀리 결합:가입)
            (인터넷 패밀리 결합:가입) -[triggers]-> (5천원 월 할인:사용)
            (T멤버십 앱:등록) -[enables]-> (1만원 할인쿠폰:다운로드)
            ```

            ## 규칙
            1. 구체적 개체명 사용 (스마트폰 X → 갤럭시 S24 ○)
            2. 중복 제거
            3. 순환 없음 (A→B→A 불가)
            4. 핵심 개체만 추출
            5. 기대 행동은 위 목록에서 선택하세요.

            메시지를 분석하여 위 형식으로 출력하세요.

                ## message:                
                {msg}


            """ # https://claude.ai/share/0354d926-8b35-42f8-935e-5e05c03e3664 에서 7번 버전
            

            # 메시지 출력
            msg_header = "==="*15+" Message "+"==="*15
            print(msg_header)
            f.write(msg_header + "\n")
            print(msg)
            f.write(msg + "\n")
            
            # DAG 출력
            dag_header = "==="*15+f" DAG ({llm_model_nm}) "+"==="*15
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

            separator = "\n" + "="*50 + "\n"
            print(separator)
            f.write(separator)

            # break
    
    print(f"출력이 파일에 저장되었습니다: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DAG 추출기')
    parser.add_argument('--num_msgs', type=int, default=50, help='추출할 메시지 수')
    parser.add_argument('--llm_model', type=str, default='ax', help='사용할 LLM 모델')
    args = parser.parse_args()
    extract_dag(num_msgs=args.num_msgs, llm_model_nm=args.llm_model)