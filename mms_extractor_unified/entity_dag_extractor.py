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
        ent, kpi = parts
        act = ""
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

# ###############################################################################
# # ─── DEMO ────────────────────────────────────────────────────────────────────
# ###############################################################################
# RAW = """
# (ADT캡스 보안 상품:구매:구매율) -[triggers]-> (무료 혜택:제공:수령율)
# (LG Q92:구매:구매율) -[triggers]-> (15% 할인:적용:할인적용율)
# (갤럭시 노트 20:구매:구매율) -[triggers]-> (중고폰 시세의 2배 보상:제공:수령율)
# (휴대폰 액정 보호 필름:방문:방문율) -[triggers]-> (무료 교체:적용:교체율)
# (가족 요금:상담:상담율) -[requires]-> (방문:방문율)
# (가정 내 인터넷:상담:상담율) -[requires]-> (방문:방문율)
# (IPTV:상담:상담율)
# """

# nodes, edges = parse_block(RAW)
# dag = build_dag(nodes, edges)

# print("\nRoot nodes (nodes with no incoming edges):")
# root_nodes = [node for node in dag.nodes() if dag.in_degree(node) == 0]
# for root in root_nodes:
#     node_data = dag.nodes[root]
#     print(f"  {root} | {node_data}")

# print("\nNodes")
# for n, d in dag.nodes(data=True):
#     print(f"  {n:30} | {d}")

# print("\nEdges")
# for u, v, d in dag.edges(data=True):
#     print(f"  {u:30} → {v:30}  ({d['relation']})")


import random

def extract_dag():

    for msg in random.sample(mms_pdf.query("msg.str.contains('')")['msg'].unique().tolist(), 50):

    #     msg = """
    # 요금제 무료혜택 안내
    # (광고)[SKT] #04 고객님, 현재 놓치고 계신 POOQ & FLO 무료 혜택을 안내해드립니다.   #91 요금제 가입 고객님은 아래 이용권 모두 무료로 이용하실 수 있어요. 다양한 방송 콘텐츠를 즐길 수 있는 POOQ과 음악을 무제한 감상할 수 있는 FLO를 무료로 즐겨보세요.  ■ POOQ 앤 데이터 (월 9,900원, 부가세 포함 → 무료) - 자세히 보기: http://t-mms.kr/t.do?m=#61&u=https://skt.sh/Dj8L4 - 지상파, 종편 실시간 TV + VOD 무제한 시청 가능 - POOQ 전용 데이터 매일 1GB 제공(전용 데이터를 다 쓰면 최대 3Mbps 속도로 계속 사용)  ■ FLO 앤 데이터 (월 7,900원, 부가세 포함 → 무료) - 자세히 보기: http://t-mms.kr/t.do?m=#61&u=https://skt.sh/l98dC - FLO 음악 무제한 듣기(모바일 기기 전용) - FLO 전용 데이터 월 3GB 제공 (음원 다운로드를 제외한 스트리밍 서비스에 한해 이용 가능)  ※ 5GX 플래티넘 요금제 가입 고객님은 POOQ 앤 데이터 플러스, FLO 앤 데이터 플러스 무료 이용 가능 (POOQ 앤 데이터/FLO 앤 데이터와 중복으로 가입할 수 없습니다.) ※ 서비스 가입 후 이용권 발급 필요 - 이용권 발급 방법: FLO 앱 > 이용권 > T 혜택 > 5GX 요금제 혜택 > 발급받기  SKT와 함께해주셔서 감사합니다.  ※ 이 메시지는 2019년 8월 19일 기준으로 작성되었습니다.  무료 수신거부 1504
        
    # """

        prompt_1 = f"""
        ## 작업
        통신사 광고 메시지에서 개체명과 기대 행동을 추출하고 DAG 형식으로 출력하세요.

        ## 출력 형식
        - 독립: `(개체명:기대행동:성과지표)`
        - 관계: `(개체명:기대행동:성과지표) -[동사구]-> (개체명:기대행동:성과지표)`

        ## 개체명 유형
        - 제품: 갤럭시 S24, 아이폰 15, 갤럭시워치
        - 서비스: 5G요금제, 인터넷, IPTV, 우주패스, FLO
        - 혜택: 50%할인, 사은품, 5만원쿠폰
        - 장소: SKT대리점, 온라인몰
        - 이벤트: 봄맞이행사, 신규가입이벤트

        ## 기대 행동
        [구매, 가입, 사용, 방문, 참여, 등록, 다운로드, 확인]

        ## 성과지표
        - 구매 → 구매율
        - 가입 → 가입율
        - 사용 → 사용율
        - 방문 → 방문율
        - 참여 → 참여율
        - 등록 → 등록율
        - 다운로드 → 다운로드율
        - 확인 → 클릭율

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
        (갤럭시 S24 Ultra:구매:구매율) -[requires]-> (5G 프라임요금제:가입:가입율)
        (5G 프라임요금제:가입:가입율) -[includes]-> (데이터 무제한:사용:사용율)
        (5G 프라임요금제:가입:가입율) -[bundles_with]-> (인터넷 패밀리 결합:가입:가입율)
        (인터넷 패밀리 결합:가입:가입율) -[triggers]-> (5천원 월 할인:사용:사용율)
        (T멤버십 앱:등록:등록율) -[enables]-> (1만원 할인쿠폰:다운로드:다운로드율)
        ```

        ## 규칙
        1. 구체적 개체명 사용 (스마트폰 X → 갤럭시 S24 ○)
        2. 중복 제거
        3. 순환 없음 (A→B→A 불가)
        4. 핵심 개체만 추출

        메시지를 분석하여 위 형식으로 출력하세요.

            ## message:                
            {msg}


        """ # https://claude.ai/share/0354d926-8b35-42f8-935e-5e05c03e3664 에서 7번 버전
        

        print("###"*15+" msg "+"###"*15)
        print(msg)
        # print("==="*15+" DAG (1) "+"==="*15)
        # dag = llm_cld.invoke(prompt_1).content
        # print(dag)

        print("==="*15+" DAG (AX4) "+"==="*15)
        dag_raw = llm_ax.invoke(prompt_1).content
        print(dag_raw)

        nodes, edges = parse_block(re.sub(r'^```|```$', '', dag_raw.strip()))
        dag = build_dag(nodes, edges)

        print("==="*15+" Root Nodes "+"==="*15)
        root_nodes = [node for node in dag.nodes() if dag.in_degree(node) == 0]
        for root in root_nodes:
            node_data = dag.nodes[root]
            print(f"  {root} | {node_data}")

        # print("==="*15+" DAG (2) "+"==="*15)
        # dag = llm_gem3.invoke(prompt_1).content
        # print(dag)



        # prompt_2 = f"""
        # 아래 DAG는 통신사 광고 메시지에서 아래 출력 형식으로 추출된 구조이다.
        # DAG에서 광고 기획자가 메시지 수신자에 실제로 원하는 '개체명:액션:성과지표'를 추출하고 싶다.
        # 루트 노드는 복수개일 수도 있다.
        # 루트 노드는 해당 광고 메시지의 성공율을 측정하는데 사용될 것이다.
        # 너무 일반적인 노드는 제외하고 추출하세요.

        # 아래 DAG에서 루트 노드를 추출하고 출력하세요.
        # 결과만 제공하세요.


        # ## DAG 형식
        # - 독립: `(개체명:기대행동:성과지표)`
        # - 관계: `(개체명:기대행동:성과지표) -[동사구]-> (개체명:기대행동:성과지표)`


        # ## DAG
        # {dag}

        # ## 메시지   
        # {msg}


        # """

        # # print("==="*15+" Core Node "+"==="*15)
        # # print(llm_gem.invoke(prompt_2).content)
        # print("==="*15+" Core Node (AX4) "+"==="*15)
        # print(llm_ax.invoke(prompt_2).content)


        print()

        break

if __name__ == "__main__":
    extract_dag()