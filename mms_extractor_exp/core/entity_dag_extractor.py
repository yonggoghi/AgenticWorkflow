"""
Entity DAG 추출기 (Entity DAG Extractor) - 엔티티 관계 그래프 분석 시스템
=====================================================================================

📋 개요
-------
이 모듈은 MMS 광고 텍스트에서 엔티티 간의 관계를 분석하여
DAG(Directed Acyclic Graph) 형태로 시각화하는 전문 도구입니다.
LLM을 활용하여 광고 내용에서 엔티티들 간의 인과관계, 순차적 액션,
의존성 등을 파악하여 구조화된 그래프로 변환합니다.

🎯 주요 기능
-----------
1. **엔티티 관계 분석**: 텍스트에서 엔티티 간 연결 관계 식별
2. **DAG 생성**: 방향성 비순환 그래프 구조 생성
3. **시각화**: NetworkX와 Graphviz를 사용한 그래프 다이어그램 생성
4. **관계 분류**: 에이전트-액션, 원인-결과, 순차적 프로세스 등 다양한 관계 타입 지원
5. **검증 및 정제**: 생성된 DAG의 유효성 검사 및 순환 참조 방지

🔧 기술 스택
-----------
- **LLM 모델**: OpenAI GPT, Anthropic Claude 등 다양한 모델 지원
- **그래프 라이브러리**: NetworkX (DAG 조작 및 검증)
- **시각화**: Graphviz (PNG/SVG 다이어그램 생성)
- **프롬프트 관리**: 외부화된 프롬프트 모듈

"""

from concurrent.futures import ThreadPoolExecutor
import time
import logging
import traceback
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from prompts.dag_extraction_prompt import build_dag_extraction_prompt
from langchain_core.output_parsers import JsonOutputParser
import json
import re
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# 로거 설정
logger = logging.getLogger(__name__)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI
from typing import List, Tuple, Union, Dict, Any, Optional, Set
import ast
from rapidfuzz import fuzz, process
import glob
import os
from config import settings
import networkx as nx
import random
from utils import create_dag_diagram, sha256_hash

llm_api_key = settings.API_CONFIG.llm_api_key
llm_api_url = settings.API_CONFIG.llm_api_url
client = OpenAI(
    api_key = llm_api_key,
    base_url = llm_api_url
)

# LLM 모델 설정
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

# 데이터 파일들을 조건부로 로드 (파일이 존재할 때만)
stop_item_names = []
mms_pdf = pd.DataFrame()

# Stop words 로드
try:
    stop_words_path = getattr(settings.METADATA_CONFIG, 'stop_items_path', './data/stop_words.csv')
    if os.path.exists(stop_words_path):
        logger.info(f"Stop words 파일 로드 중: {stop_words_path}")
        stop_item_names = pd.read_csv(stop_words_path)['stop_words'].to_list()
        logger.info(f"Stop words 로드 완료: {len(stop_item_names)}개")
    else:
        logger.warning(f"Stop words 파일을 찾을 수 없습니다: {stop_words_path}")
except Exception as e:
    logger.warning(f"Stop words 파일 로드 실패: {e}")
    stop_item_names = []

# MMS 메시지 데이터 로드
try:
    mms_msg_path = getattr(settings.METADATA_CONFIG, 'mms_msg_path', './data/mms_messages.csv')
    
    if os.path.exists(mms_msg_path):
        logger.info(f"MMS 데이터 파일 로드 중: {mms_msg_path}")
        mms_pdf = pd.read_csv(mms_msg_path)
        logger.info(f"MMS 데이터 원본 크기: {mms_pdf.shape}")
        logger.info(f"MMS 데이터 컬럼들: {list(mms_pdf.columns)}")
        
        # 컬럼명 확인 및 표준화
        if 'msg' not in mms_pdf.columns:
            # 1. 대소문자 구분 없이 msg 컬럼 찾기
            msg_col_candidates = [col for col in mms_pdf.columns if col.lower() == 'msg']
            if msg_col_candidates:
                logger.info(f"'msg' 컬럼을 '{msg_col_candidates[0]}'로 리네임")
                mms_pdf = mms_pdf.rename(columns={msg_col_candidates[0]: 'msg'})
            # 2. mms_phrs 컬럼 확인 (일반적인 MMS 메시지 컬럼명)
            elif 'mms_phrs' in mms_pdf.columns:
                logger.info("'mms_phrs' 컬럼을 'msg'로 리네임")
                mms_pdf = mms_pdf.rename(columns={'mms_phrs': 'msg'})
            # 3. MMS_PHRS 컬럼 확인 (대문자 버전)
            elif 'MMS_PHRS' in mms_pdf.columns:
                logger.info("'MMS_PHRS' 컬럼을 'msg'로 리네임")
                mms_pdf = mms_pdf.rename(columns={'MMS_PHRS': 'msg'})
            # 4. msg_nm 컬럼 확인 (메시지 이름)
            elif 'msg_nm' in mms_pdf.columns:
                logger.info("'msg_nm' 컬럼을 'msg'로 리네임")
                mms_pdf = mms_pdf.rename(columns={'msg_nm': 'msg'})
            else:
                logger.warning("'msg' 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼들:")
                logger.warning(f"{list(mms_pdf.columns)}")
                # 빈 DataFrame으로 설정
                mms_pdf = pd.DataFrame()
        
        # 문자열 타입으로 변환
        if 'msg' in mms_pdf.columns:
            mms_pdf['msg'] = mms_pdf['msg'].astype('str')
            logger.info(f"'msg' 컬럼을 문자열 타입으로 변환 완료")
            
            # 데이터 품질 확인
            null_count = mms_pdf['msg'].isnull().sum()
            empty_count = (mms_pdf['msg'] == '').sum()
            valid_count = len(mms_pdf) - null_count - empty_count
            logger.info(f"MMS 메시지 품질: 유효={valid_count}, 빈값={empty_count}, null={null_count}")
            
            # 샘플 데이터 확인
            if not mms_pdf.empty and valid_count > 0:
                sample_msgs = mms_pdf['msg'].dropna().head(2).tolist()
                logger.info(f"MMS 메시지 샘플: {[msg[:50]+'...' if len(msg) > 50 else msg for msg in sample_msgs]}")
        
        logger.info(f"MMS 데이터 로드 완료: {len(mms_pdf)}개 행")
    else:
        logger.warning(f"MMS 데이터 파일을 찾을 수 없습니다: {mms_msg_path}")
        logger.warning("샘플 메시지로 테스트하려면 --prompt_mode simple 옵션을 사용하세요")
        mms_pdf = pd.DataFrame()
        
except Exception as e:
    logger.error(f"MMS 데이터 파일 로드 실패: {e}")
    logger.error(f"오류 상세: {traceback.format_exc()}")
    mms_pdf = pd.DataFrame()  # 빈 DataFrame으로 초기화

###############################################################################
# 1) 기존 정규식 및 파서 (하위 호환성 유지)
###############################################################################
PAT = re.compile(r"\((.*?)\)\s*-\[(.*?)\]->\s*\((.*?)\)")
NODE_ONLY = re.compile(r"\((.*?)\)\s*$")

def parse_dag_block(text: str):
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
# 2) 개선된 노드 스플리터 – 유연한 파트 처리
###############################################################################
def split_node_parts(raw: str) -> Dict[str,str]:
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
# 3) 기존 DAG 빌더 (하위 호환성 유지)
###############################################################################
def build_dag_graph(nodes: List[str], edges: List[Tuple[str,str,str]]) -> nx.DiGraph:
    g = nx.DiGraph()
    
    # Add nodes with error handling
    for n in nodes:
        try:
            node_attrs = split_node_parts(n)
            g.add_node(n, **node_attrs)
        except ValueError as e:
            print(f"Warning: {e}")
            # Add node with minimal attributes if parsing fails
            g.add_node(n, entity=n, action="", metric="")
    
    # Add edges
    for src, dst, rel in edges:
        g.add_edge(src, dst, relation=rel)
    
    return g

###############################################################################
# 4) 기존 Path Finder (하위 호환성 유지)
###############################################################################
def extract_root_to_leaf_paths(dag):
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

###############################################################################
# 5) 새로운 개선된 DAGParser 클래스
###############################################################################
class DAGParser:
    """
    DAG 파싱 클래스
    
    LLM이 생성한 DAG 텍스트를 NetworkX 그래프 객체로 변환합니다.
    
    주요 기능:
    - LLM 응답에서 DAG 섹션 추출
    - DAG 텍스트를 NetworkX DiGraph로 파싱
    - 노드와 엣지 관계 분석
    
    지원하는 DAG 형식:
    - 엣지: (엔티티:행동) -[관계동사]-> (엔티티:행동)
    - 독립 노드: (엔티티:행동)
    """
    
    def __init__(self):
        # 개선된 정규표현식 패턴 - 관계 부분에 쉼표와 공백 허용
        # 관계 부분([...])에 모든 문자 허용 (]를 제외하고)
        self.edge_pattern = r'\(([^:)]+):([^)]+)\)\s*-\[([^\]]+)\]->\s*\(([^:)]+):([^)]+)\)'
        # 독립형 노드 패턴 추가
        self.standalone_node_pattern = r'\(([^:)]+):([^)]+)\)\s*$'
        # 섹션 패턴 수정: ## 또는 ###로 시작하는 2. 추출된 DAG 섹션
        self.section_pattern = r'#{2,3}\s*2\.\s*추출된\s*DAG'
        
    def parse_dag_line(self, line: str) -> Optional[Union[Tuple[str, str, str, str, str], Tuple[str, str]]]:
        """단일 DAG 라인을 파싱하여 구성 요소 반환"""
        # 먼저 엣지 패턴 확인
        edge_match = re.match(self.edge_pattern, line)
        if edge_match:
            return (
                edge_match.group(1).strip(),  # src_entity
                edge_match.group(2).strip(),  # src_action
                edge_match.group(3).strip(),  # relation (쉼표, 조건 포함 가능)
                edge_match.group(4).strip(),  # dst_entity
                edge_match.group(5).strip()   # dst_action
            )
        
        # 독립형 노드 패턴 확인
        standalone_match = re.match(self.standalone_node_pattern, line)
        if standalone_match:
            return (
                standalone_match.group(1).strip(),  # entity
                standalone_match.group(2).strip()   # action
            )
        
        return None
        
    def extract_dag_section(self, full_text: str) -> str:
        """전체 텍스트에서 DAG 섹션만 추출"""
        lines = full_text.split('\n')
        
        # 더 유연한 DAG 섹션 찾기
        dag_section_patterns = [
            r'#{2,3}\s*최종\s*DAG',            # 최종 DAG
            r'#{2,3}\s*4\.\s*수정된\s*DAG',    # 4. 수정된 DAG 최우선
            r'#{2,3}\s*수정된\s*DAG',          # 수정된 DAG
            r'#{2,3}\s*2\.\s*추출된\s*DAG',    # 2. 추출된 DAG (기본)
            r'#{2,3}\s*DAG',
            r'추출된\s*DAG',
            r'2\.\s*추출된\s*DAG'
        ]
        
        # DAG 섹션 찾기
        start_idx = -1
        end_idx = len(lines)
        in_dag_section = False
        in_code_block = False
        
        # 패턴들을 순차적으로 시도
        for pattern in dag_section_patterns:
            for i, line in enumerate(lines):
                if re.search(pattern, line, re.IGNORECASE):
                    in_dag_section = True
                    
                    # DAG 헤더 다음에서 ``` 찾기 (빈 줄이 있어도 괜찮음)
                    code_block_found = False
                    for j in range(i + 1, min(i + 4, len(lines))):  # 최대 3줄까지 확인
                        next_line = lines[j].strip()
                        if next_line == '```':
                            start_idx = j + 1
                            in_code_block = True
                            code_block_found = True
                            break
                        elif next_line and not next_line == '':  # 빈 줄이 아닌 다른 내용이 나오면 중단
                            # DAG 패턴으로 시작하는 줄이면 코드블록 없이 시작
                            if (re.match(self.edge_pattern, next_line) or 
                                re.match(self.standalone_node_pattern, next_line)):
                                start_idx = j
                                in_code_block = False
                                code_block_found = True
                                break
                            else:
                                break
                    
                    # 코드블록을 찾지 못했다면 헤더 다음 줄부터 시작
                    if not code_block_found:
                        start_idx = i + 1
                        in_code_block = False
                    
                    break
            if start_idx != -1:
                break
        
        # DAG 섹션 종료 조건 찾기
        if start_idx != -1:
            for i in range(start_idx, len(lines)):
                line = lines[i]
                if in_code_block and line.strip() == '```':
                    end_idx = i
                    break
                elif not in_code_block and re.match(r'#{2,3}\s*[3-9]\.', line):  # 3번 이상 섹션에서 종료
                    end_idx = i
                    break
        
        if start_idx == -1:
            # 섹션 헤더가 없는 경우, DAG 패턴을 직접 찾기
            dag_lines = []
            for line in lines:
                if (re.match(self.edge_pattern, line) or 
                    re.match(self.standalone_node_pattern, line) or 
                    line.strip().startswith('#')):
                    dag_lines.append(line)
            
            if dag_lines:
                result = '\n'.join(dag_lines)
                return result
            else:
                raise ValueError("DAG 섹션을 찾을 수 없습니다.")
        
        result = '\n'.join(lines[start_idx:end_idx])
        return result
    
    def parse_dag(self, dag_text: str) -> nx.DiGraph:
        """DAG 텍스트를 NetworkX DiGraph로 변환"""
        G = nx.DiGraph()
        
        # 통계 정보 저장
        stats = {
            'total_edges': 0,
            'comment_lines': 0,
            'empty_lines': 0,
            'paths': [],
            'parse_errors': [],
            'parsed_lines': []
        }
        
        current_path = None
        
        for line_num, line in enumerate(dag_text.strip().split('\n'), 1):
            line = line.strip()
            
            # 빈 라인 처리
            if not line:
                stats['empty_lines'] += 1
                continue
            
            # 주석 라인 처리 (경로 정보 추출)
            if line.startswith('#'):
                stats['comment_lines'] += 1
                current_path = line[1:].strip()
                if current_path:
                    stats['paths'].append(current_path)
                continue
            
            # DAG 엣지 또는 독립형 노드 파싱
            parsed = self.parse_dag_line(line)
            if parsed:
                try:
                    if len(parsed) == 5:  # 엣지 (src_entity, src_action, relation, dst_entity, dst_action)
                        src_entity, src_action, relation, dst_entity, dst_action = parsed
                        
                        # 노드 ID 생성
                        src_node = f"{src_entity}:{src_action}"
                        dst_node = f"{dst_entity}:{dst_action}"
                        
                        # 노드 추가 (속성 포함)
                        G.add_node(src_node, 
                                  entity=src_entity, 
                                  action=src_action,
                                  path=current_path)
                        G.add_node(dst_node, 
                                  entity=dst_entity, 
                                  action=dst_action,
                                  path=current_path)
                        
                        # 엣지 추가 (관계에 쉼표나 조건이 포함될 수 있음)
                        G.add_edge(src_node, dst_node, 
                                  relation=relation,
                                  path=current_path)
                        
                        stats['total_edges'] += 1
                        stats['parsed_lines'].append(f"Line {line_num}: {src_node} -[{relation}]-> {dst_node}")
                        
                    elif len(parsed) == 2:  # 독립형 노드 (entity, action)
                        entity, action = parsed
                        
                        # 노드 ID 생성
                        node_id = f"{entity}:{action}"
                        
                        # 독립형 노드 추가
                        G.add_node(node_id, 
                                  entity=entity, 
                                  action=action,
                                  path=current_path)
                        
                        stats['parsed_lines'].append(f"Line {line_num}: Standalone node {node_id}")
                    
                except Exception as e:
                    stats['parse_errors'].append(f"Line {line_num}: {str(e)}")
            else:
                # 파싱 실패한 라인 기록 (주석이 아닌 경우만)
                if not line.startswith('#') and line.strip():
                    stats['parse_errors'].append(f"Line {line_num}: 패턴 매칭 실패 - {line[:80]}...")
        
        # 그래프에 통계 정보 저장
        G.graph['stats'] = stats
        
        return G
    
    def get_root_nodes(self, G: nx.DiGraph) -> List[str]:
        """Root 노드(들어오는 엣지가 없는 노드) 찾기"""
        return [node for node in G.nodes() if G.in_degree(node) == 0]
    
    def get_leaf_nodes(self, G: nx.DiGraph) -> List[str]:
        """Leaf 노드(나가는 엣지가 없는 노드) 찾기"""
        return [node for node in G.nodes() if G.out_degree(node) == 0]
    
    def get_paths_from_root_to_leaf(self, G: nx.DiGraph) -> List[List[str]]:
        """Root에서 Leaf까지의 모든 경로 찾기"""
        roots = self.get_root_nodes(G)
        leaves = self.get_leaf_nodes(G)
        
        all_paths = []
        for root in roots:
            for leaf in leaves:
                try:
                    paths = list(nx.all_simple_paths(G, root, leaf))
                    all_paths.extend(paths)
                except nx.NetworkXNoPath:
                    continue
        
        return all_paths
    
    def analyze_graph(self, G: nx.DiGraph) -> Dict:
        """그래프 분석 정보 생성"""
        analysis = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'root_nodes': self.get_root_nodes(G),
            'leaf_nodes': self.get_leaf_nodes(G),
            'is_dag': nx.is_directed_acyclic_graph(G),
            'num_components': nx.number_weakly_connected_components(G),
            'paths_info': G.graph.get('stats', {}).get('paths', []),
            'longest_path_length': 0
        }
        
        # 최장 경로 찾기
        if analysis['is_dag'] and G.number_of_nodes() > 0:
            try:
                longest = nx.dag_longest_path(G)
                analysis['longest_path_length'] = len(longest) - 1 if longest else 0
            except:
                analysis['longest_path_length'] = 0
        
        return analysis
    
    def to_json(self, G: nx.DiGraph) -> str:
        """그래프를 JSON 형식으로 변환"""
        data = {
            'nodes': [
                {
                    'id': node,
                    'entity': G.nodes[node].get('entity', ''),
                    'action': G.nodes[node].get('action', ''),
                    'path': G.nodes[node].get('path', '')
                }
                for node in G.nodes()
            ],
            'edges': [
                {
                    'source': edge[0],
                    'target': edge[1],
                    'relation': G.edges[edge].get('relation', ''),
                    'path': G.edges[edge].get('path', '')
                }
                for edge in G.edges()
            ],
            'analysis': self.analyze_graph(G)
        }
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def visualize_paths(self, G: nx.DiGraph) -> str:
        """경로별로 구조화된 텍스트 출력"""
        output = []
        paths_dict = {}
        
        # 경로별로 엣지 그룹화
        for edge in G.edges():
            path = G.edges[edge].get('path', 'Unknown')
            if path not in paths_dict:
                paths_dict[path] = []
            paths_dict[path].append(edge)
        
        # 경로별 출력
        for path, edges in paths_dict.items():
            if path and path != 'Unknown':
                output.append(f"\n[{path}]")
            for edge in edges:
                relation = G.edges[edge].get('relation', '')
                output.append(f"  {edge[0]} -{relation}-> {edge[1]}")
        
        return '\n'.join(output)


def build_dag_from_ontology(ont_result: dict) -> nx.DiGraph:
    """
    ONT/KG 결과에서 NetworkX DiGraph 생성 (LLM 호출 없음)

    KG 모드에서 Step 7의 KG 메타데이터를 Step 11의 DAG 시각화로 변환할 때 사용.

    Args:
        ont_result: ONT 모드에서 추출된 결과
            {
                'dag_text': str,
                'entity_types': dict,  # {entity_id: type}
                'relationships': list  # [{source, target, type}, ...]
            }

    Returns:
        nx.DiGraph: DAG 그래프
    """
    G = nx.DiGraph()

    entity_types = ont_result.get('entity_types', {})
    relationships = ont_result.get('relationships', [])
    dag_text = ont_result.get('dag_text', '')

    # 방법 1: relationships에서 그래프 생성 (더 정확한 타입 정보 보존)
    if relationships:
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            src = rel.get('source', '')
            tgt = rel.get('target', '')
            rel_type = rel.get('type', '')

            if src and tgt:
                # 노드 추가 (타입 정보 포함)
                src_type = entity_types.get(src, 'Unknown')
                tgt_type = entity_types.get(tgt, 'Unknown')

                # 노드 ID에 타입 포함 (예: "9월 T day:Campaign")
                src_node_id = f"{src}:{src_type}"
                tgt_node_id = f"{tgt}:{tgt_type}"

                G.add_node(src_node_id, entity=src, entity_type=src_type, action='')
                G.add_node(tgt_node_id, entity=tgt, entity_type=tgt_type, action='')

                # 엣지 추가
                G.add_edge(src_node_id, tgt_node_id, relation=rel_type)

        logger.info(f"📊 ONT 그래프 생성 (relationships 기반): {G.number_of_nodes()} 노드, {G.number_of_edges()} 엣지")
        return G

    # 방법 2: dag_text 파싱 (relationships가 없는 경우)
    if dag_text:
        # DAG 패턴: (Entity:Action) -[Relation]-> (Entity:Action)
        dag_pattern = r'\(([^:)]+):([^)]+)\)\s*-\[([^\]]+)\]->\s*\(([^:)]+):([^)]+)\)'
        matches = re.findall(dag_pattern, dag_text)

        for match in matches:
            src_entity, src_action, relation, dst_entity, dst_action = match

            src_node = f"{src_entity.strip()}:{src_action.strip()}"
            dst_node = f"{dst_entity.strip()}:{dst_action.strip()}"

            src_type = entity_types.get(src_entity.strip(), 'Unknown')
            dst_type = entity_types.get(dst_entity.strip(), 'Unknown')

            G.add_node(src_node, entity=src_entity.strip(), action=src_action.strip(), entity_type=src_type)
            G.add_node(dst_node, entity=dst_entity.strip(), action=dst_action.strip(), entity_type=dst_type)
            G.add_edge(src_node, dst_node, relation=relation.strip())

        logger.info(f"📊 ONT 그래프 생성 (dag_text 파싱): {G.number_of_nodes()} 노드, {G.number_of_edges()} 엣지")

    return G


def extract_dag(parser: DAGParser, msg: str, llm_model, prompt_mode: str = 'cot'):
    """
    엔티티 관계 DAG 추출 메인 함수
    ========================================
    
    🎯 목적
    -------
    MMS 광고 텍스트에서 엔티티 간의 복잡한 관계를 분석하여
    방향성 비순환 그래프(DAG) 형태로 시각화 가능한 구조로 변환합니다.
    
    🔄 처리 과정
    -----------
    1. **LLM 기반 관계 추출**: 전문 프롬프트를 사용하여 엔티티 간 관계 식별
    2. **구조화된 파싱**: 자연어 설명에서 DAG 섹션 추출 및 정리
    3. **그래프 변환**: 정규표현식 기반 파싱으로 NetworkX 그래프 생성
    4. **검증 및 정제**: DAG 유효성 검사 및 순환 참조 방지
    
    📊 출력 데이터
    -----------
    - **dag_section**: 파싱된 DAG 텍스트 (인간 가독)
    - **dag**: NetworkX DiGraph 객체 (프로그래밍 활용)
    - **dag_raw**: LLM 원본 응답 (디버깅 용도)
    
    Args:
        parser (DAGParser): DAG 파싱 전문 객체
        msg (str): 분석할 MMS 메시지 텍스트
        llm_model: Langchain 호환 LLM 모델 인스턴스
        prompt_mode (str): 프롬프트 모드 ('cot' 또는 'simple'). 기본값 'cot'.
        
    Returns:
        dict: DAG 추출 결과
            {
                'dag_section': str,      # 구조화된 DAG 텍스트
                'dag': nx.DiGraph,       # NetworkX 그래프 객체
                'dag_raw': str,          # LLM 원본 응답
                'nodes': List[str],      # 추출된 노드 목록
                'edges': List[Tuple],    # 추출된 엣지 목록
            }
    
    Raises:
        Exception: LLM API 호출 실패, 파싱 오류 등
        
    Example:
        >>> parser = DAGParser()
        >>> result = extract_dag(parser, "SK텔레콤 혜택 안내...", llm_model, prompt_mode='simple')
        >>> print(f"DAG 노드 수: {result['dag'].number_of_nodes()}")
        >>> print(f"DAG 엣지 수: {result['dag'].number_of_edges()}")
    """
    
    # 초기 로깅 및 상태 설정
    logger.info("🚀 DAG 추출 프로세스 시작")
    logger.info(f"📝 입력 메시지 길이: {len(msg)}자")
    logger.info(f"🤖 사용 LLM 모델: {llm_model}")
    logger.info(f"⚙️  프롬프트 모드: {prompt_mode}")
    
    # 단계 1: 외부 프롬프트 모듈에서 전문 프롬프트 구성
    prompt = build_dag_extraction_prompt(msg, mode=prompt_mode)
    
    # LLM 호출 준비 로깅
    logger.info("🤖 LLM에 DAG 추출 요청 중...")
    logger.info(f"📏 프롬프트 길이: {len(prompt)}자")
    
    # 단계 2: LLM 호출을 통한 엔티티 간 관계 분석
    try:
        # 프롬프트 저장 (디버깅/미리보기용)
        if hasattr(llm_model, '_store_prompt_for_preview'):
            llm_model._store_prompt_for_preview(prompt, "dag_extraction")
        else:
            # 전역 프롬프트 저장소 사용
            import threading
            if not hasattr(threading.current_thread(), 'stored_prompts'):
                threading.current_thread().stored_prompts = {}
            threading.current_thread().stored_prompts['dag_extraction_prompt'] = {
                'title': 'DAG 관계 추출 프롬프트',
                'description': '엔티티 간의 관계를 그래프 형태로 추출하는 프롬프트',
                'content': prompt,
                'length': len(prompt)
            }
        
        dag_raw = llm_model.invoke(prompt).content
        logger.info(f"📝 LLM 응답 길이: {len(dag_raw)}자")
        logger.info(f"📄 LLM 응답 미리보기 (처음 500자): {dag_raw[:500]}...")
        print("\n" + "="*80)
        print("🔍 [DEBUG] LLM 전체 응답:")
        print("="*80)
        print(dag_raw)
        print("="*80 + "\n")
    except Exception as e:
        logger.error(f"❌ LLM 호출 중 오류 발생: {e}")
        raise

    # Step 2: DAG 섹션 추출 및 정리
    # LLM 응답에서 실제 DAG 구조 부분만 추출
    logger.info("🔍 DAG 섹션 추출 중...")
    try:
        dag_section = parser.extract_dag_section(dag_raw)
        logger.info(f"📄 추출된 DAG 섹션 길이: {len(dag_section)}자")
        if dag_section:
            logger.info(f"📄 DAG 섹션 내용:\n{dag_section}")
        else:
            logger.warning("⚠️ DAG 섹션이 비어있습니다")
    except Exception as e:
        logger.error(f"❌ DAG 섹션 추출 실패: {e}")
        logger.error(f"❌ LLM 응답 전체:\n{dag_raw}")
        raise
    
    # Step 3: NetworkX 그래프 구조 생성
    # 텍스트 DAG를 실제 그래프 객체로 변환
    logger.info("🔗 DAG 파싱 중...")
    dag = parser.parse_dag(dag_section)
    logger.info(f"📊 파싱된 DAG - 노드 수: {dag.number_of_nodes()}, 엣지 수: {dag.number_of_edges()}")
    
    # Step 4: 결과 검증 및 로깅
    if dag.number_of_nodes() > 0:
        logger.info(f"🎯 DAG 노드 목록: {list(dag.nodes())}")
        logger.info(f"🔗 DAG 엣지 목록: {list(dag.edges())}")
    else:
        logger.warning("⚠️ DAG에 노드가 없습니다")

    logger.info("✅ DAG 추출 프로세스 완료")
    
    # 결과 딕셔너리 반환
    return {
        'dag_section': dag_section,  # 텍스트 형태의 DAG 표현
        'dag': dag,                  # NetworkX DiGraph 객체
        'dag_raw': dag_raw           # LLM 원본 응답 (디버깅용)
    }

    # root_nodes = [node for node in dag.nodes() if dag.in_degree(node) == 0]
    # for root in root_nodes:
    #     node_data = dag.nodes[root]
    #     print(f"  {root} | {node_data}")

    # paths, roots, leaves = get_root_to_leaf_paths(dag)

    # for i, path in enumerate(paths):
    #     print(f"\nPath {i+1}:")
    #     for j, node in enumerate(path):
    #         if j < len(path) - 1:
    #             edge_data = dag.get_edge_data(node, path[j+1])
    #             relation = edge_data['relation'] if edge_data else ''
    #             print(f"  {node}")
    #             print(f"    --[{relation}]-->")
    #         else:
    #             print(f"  {node}")
                

###############################################################################
# 7) 메인 추출 함수 (기존 인터페이스 유지 + 개선 기능 추가)
###############################################################################
def dag_finder(num_msgs=50, llm_model_nm='ax', save_dag_image=True, prompt_mode='cot'):

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

    # 데이터 검증
    if mms_pdf is None or mms_pdf.empty or 'msg' not in mms_pdf.columns:
        logger.warning("MMS 데이터가 로드되지 않았습니다. 샘플 메시지로 테스트합니다.")
        sample_messages = [
            "[SKT] T 우주패스 쇼핑 출시! 지금 링크를 눌러 가입하면 첫 달 1,000원에 이용 가능합니다. 가입 고객 전원에게 11번가 포인트 3,000P와 아마존 무료배송 쿠폰을 드립니다.",
            "[SKT] 에스알대리점 지행점 9월 특가. 아이폰16 즉시 개통 가능! 매장 방문하셔서 상담만 받아도 사은품을 드립니다. 위치: 지행역 2번 출구"
        ]
        messages_to_process = sample_messages[:min(num_msgs, len(sample_messages))]
    else:
        # 기존 로직: mms_pdf에서 랜덤 샘플링
        try:
            all_msgs = mms_pdf['msg'].unique().tolist()
            messages_to_process = random.sample(all_msgs, min(num_msgs, len(all_msgs)))
        except Exception as e:
            logger.error(f"메시지 샘플링 중 오류: {e}")
            return

    # 출력을 파일에 저장하기 위한 설정
    output_file = "./logs/dag_extraction_output.txt"

    line_break_patterns = {"__":"\n", "■":"\n■", "▶":"\n▶", "_":"\n"}
    
    # 개선된 파서 초기화
    parser = DAGParser()
    dag = None  # dag 변수 초기화
    
    with open(output_file, 'a', encoding='utf-8') as f:
        # 실행 시작 시점 기록
        from datetime import datetime
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'='*80}\n")
        f.write(f"DAG 추출 실행 시작: {start_time}\n")
        f.write(f"설정: 모델={llm_model_nm}, 모드={prompt_mode}\n")
        f.write(f"{'='*80}\n\n")
        
        for msg in messages_to_process:
            dag = None  # 각 메시지마다 dag 초기화
            try:
                for pattern, replacement in line_break_patterns.items():
                    msg = msg.replace(pattern, replacement)
                
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
                
                print(f"🚀 extract_dag 함수 호출 중... (prompt_mode={prompt_mode})")
                extract_dag_result = extract_dag(parser, msg, llm_model, prompt_mode=prompt_mode)

                dag_raw = extract_dag_result['dag_raw']
                dag_section = extract_dag_result['dag_section']
                dag = extract_dag_result['dag']

                print("\n" + "="*80)
                print("📄 LLM 원본 응답 (dag_raw):")
                print("="*80)
                print(dag_raw)
                print("="*80 + "\n")
                f.write(dag_raw + "\n")

                # 파서 선택 및 처리
                if parser:
                    try:                    
                        # 디버깅을 위해 dag_section 내용 확인
                        print("=== DAG Section Debug ===")
                        print(f"DAG Section Length: {len(dag_section)}")
                        print("DAG Section Content:")
                        print(dag_section)
                        print("=" * 50)

                        # 라인별로 확인
                        lines = dag_section.strip().split('\n')
                        print(f"Total lines: {len(lines)}")
                        for i, line in enumerate(lines, 1):
                            line = line.strip()
                            if line:
                                print(f"Line {i}: '{line}'")
                        print("=" * 50)

                        # 파싱 과정 디버깅
                        dag = parser.parse_dag(dag_section)

                        # 파싱 결과 확인
                        print("=== Parse Results ===")
                        print(f"Nodes: {dag.number_of_nodes()}")
                        print(f"Edges: {dag.number_of_edges()}")

                        # 파싱 통계 확인
                        if dag.graph.get('stats'):
                            stats = dag.graph['stats']
                            print(f"Parsed lines: {len(stats['parsed_lines'])}")
                            print(f"Parse errors: {len(stats['parse_errors'])}")
                            
                            if stats['parse_errors']:
                                print("Parse Errors:")
                                for error in stats['parse_errors']:
                                    print(f"  - {error}")
                                    
                            if stats['parsed_lines']:
                                print("Successfully parsed lines:")
                                for parsed in stats['parsed_lines']:
                                    print(f"  + {parsed}")
                        
                        # 분석 정보 출력
                        analysis = parser.analyze_graph(dag)
                        analysis_header = "==="*15+" Enhanced Analysis "+"==="*15
                        print(analysis_header)
                        f.write(analysis_header + "\n")
                        
                        analysis_info = f"""그래프 분석:
- 노드 수: {analysis['num_nodes']}
- 엣지 수: {analysis['num_edges']}  
- Root 노드: {analysis['root_nodes']}
- Leaf 노드: {analysis['leaf_nodes']}
- DAG 여부: {analysis['is_dag']}
- 최장 경로 길이: {analysis['longest_path_length']}"""
                        print(analysis_info)
                        f.write(analysis_info + "\n")
                        
                        # 파싱 에러가 있다면 출력
                        if dag.graph['stats'].get('parse_errors'):
                            error_info = "\n파싱 에러:"
                            print(error_info)
                            f.write(error_info + "\n")
                            for error in dag.graph['stats']['parse_errors'][:3]:  # 처음 3개만
                                error_line = f"  ✗ {error}"
                                print(error_line)
                                f.write(error_line + "\n")
                                
                    except Exception as e:
                        print(f"Enhanced parser 실패, 기본 파서로 전환: {e}")
                        f.write(f"Enhanced parser 실패, 기본 파서로 전환: {e}\n")
                        # 기본 파서로 폴백
                        nodes, edges = parse_dag_block(re.sub(r'^```|```$', '', dag_raw.strip()))
                        dag = build_dag_graph(nodes, edges)
                else:
                    # 기본 파서 사용
                    nodes, edges = parse_dag_block(re.sub(r'^```|```$', '', dag_raw.strip()))
                    dag = build_dag_graph(nodes, edges)

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
                paths, roots, leaves = extract_root_to_leaf_paths(dag)
                
                if not paths:
                    print("No paths found.")
                    f.write("No paths found.\n")

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

    if save_dag_image and dag is not None:
        try:
            create_dag_diagram(dag, filename=f'dag_#_{sha256_hash(msg)}')
            print(f"DAG 이미지가 저장되었습니다: {f'dag_#_{sha256_hash(msg)}.png'}")
        except Exception as e:
            print(f"DAG 이미지 저장 실패: {e}")
    elif save_dag_image and dag is None:
        print("⚠️ DAG 객체가 생성되지 않아 이미지를 저장할 수 없습니다.")

if __name__ == "__main__":
    import argparse
    
    parser_arg = argparse.ArgumentParser(description='DAG 추출기 - MMS 메시지에서 엔티티 관계 그래프 추출')
    parser_arg.add_argument('--message', type=str, help='단일 메시지 직접 입력')
    parser_arg.add_argument('--batch-file', type=str, help='배치 처리할 메시지가 담긴 파일 경로 (한 줄에 하나씩)')
    parser_arg.add_argument('--num_msgs', type=int, default=50, help='CSV에서 추출할 메시지 수 (기본값: 50)')
    parser_arg.add_argument('--llm_model', type=str, default='ax', help='사용할 LLM 모델 (기본값: ax)')
    parser_arg.add_argument('--save_dag_image', action='store_true', default=False, help='DAG 이미지 저장 여부')
    parser_arg.add_argument('--prompt_mode', type=str, default='cot', choices=['cot', 'simple'], help='프롬프트 모드 (cot: Chain-of-Thought 상세분석, simple: 간단분석)')
    args = parser_arg.parse_args()

    args.message = """
  message: '(광고)[SKT] iPhone 신제품 구매 혜택 안내 __#04 고객님, 안녕하세요._SK텔레콤에서 iPhone 신제품 구매하면, 최대 22만 원 캐시백 이벤트에 참여하실 수 있습니다.__현대카드로 애플 페이도 더 편리하게 이용해 보세요.__▶ 현대카드 바로 가기: https://t-mms.kr/ais/#74_ _애플 페이 티머니 충전 쿠폰 96만 원, 샌프란시스코 왕복 항공권, 애플 액세서리 팩까지!_Lucky 1717 이벤트 응모하고 경품 당첨의 행운을 누려 보세요.__▶ 이벤트 자세히 보기: https://t-mms.kr/aiN/#74_ _■ 문의: SKT 고객센터(1558, 무료)__SKT와 함께해 주셔서 감사합니다.__무료 수신거부 1504',

    """
    
    # 단일 메시지 처리
    if args.message:
        print("=" * 80)
        print("🚀 단일 메시지 DAG 추출 시작")
        print("=" * 80)
        print(f"메시지: {args.message[:100]}..." if len(args.message) > 100 else f"메시지: {args.message}")
        print(f"LLM 모델: {args.llm_model}")
        print(f"프롬프트 모드: {args.prompt_mode}")
        print("=" * 80 + "\n")
        
        # LLM 모델 초기화
        if args.llm_model == 'ax':
            llm_model = llm_ax
        elif args.llm_model == 'gem':
            llm_model = llm_gem
        elif args.llm_model == 'cld':
            llm_model = llm_cld
        elif args.llm_model == 'gen':
            llm_model = llm_gen
        elif args.llm_model == 'gpt':
            llm_model = llm_gpt
        else:
            llm_model = llm_ax
        
        # DAG 추출
        parser = DAGParser()
        try:
            result = extract_dag(parser, args.message, llm_model, prompt_mode=args.prompt_mode)
            
            print("\n" + "=" * 80)
            print("✅ DAG 추출 완료")
            print("=" * 80)
            print(f"추출된 DAG:\n{result['dag_section']}")
            print("=" * 80)
            print(f"노드 수: {result['dag'].number_of_nodes()}")
            print(f"엣지 수: {result['dag'].number_of_edges()}")
            
            if args.save_dag_image and result['dag'].number_of_nodes() > 0:
                dag_filename = f"dag_#_{sha256_hash(args.message)}"
                create_dag_diagram(result['dag'], filename=dag_filename)
                print(f"✅ DAG 이미지 저장: {dag_filename}.png")
                
        except Exception as e:
            print(f"❌ DAG 추출 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 배치 파일 처리
    elif args.batch_file:
        print("=" * 80)
        print("🚀 배치 파일 DAG 추출 시작")
        print("=" * 80)
        print(f"파일: {args.batch_file}")
        print(f"LLM 모델: {args.llm_model}")
        print(f"프롬프트 모드: {args.prompt_mode}")
        print("=" * 80 + "\n")
        
        try:
            with open(args.batch_file, 'r', encoding='utf-8') as f:
                messages = [line.strip() for line in f if line.strip()]
            
            print(f"📄 로드된 메시지 수: {len(messages)}개\n")
            
            # LLM 모델 초기화
            if args.llm_model == 'ax':
                llm_model = llm_ax
            elif args.llm_model == 'gem':
                llm_model = llm_gem
            elif args.llm_model == 'cld':
                llm_model = llm_cld
            elif args.llm_model == 'gen':
                llm_model = llm_gen
            elif args.llm_model == 'gpt':
                llm_model = llm_gpt
            else:
                llm_model = llm_ax
            
            parser = DAGParser()
            
            for idx, msg in enumerate(messages, 1):
                print(f"\n{'='*80}")
                print(f"처리 중: {idx}/{len(messages)}")
                print(f"메시지: {msg[:100]}..." if len(msg) > 100 else f"메시지: {msg}")
                print('='*80)
                
                try:
                    result = extract_dag(parser, msg, llm_model, prompt_mode=args.prompt_mode)
                    print(f"✅ 노드: {result['dag'].number_of_nodes()}개, 엣지: {result['dag'].number_of_edges()}개")
                    
                    if args.save_dag_image and result['dag'].number_of_nodes() > 0:
                        dag_filename = f"dag_batch_{idx}_{sha256_hash(msg)}"
                        create_dag_diagram(result['dag'], filename=dag_filename)
                        print(f"✅ 이미지 저장: {dag_filename}.png")
                        
                except Exception as e:
                    print(f"❌ 실패: {e}")
            
            print(f"\n{'='*80}")
            print(f"✅ 배치 처리 완료: {len(messages)}개 메시지")
            print('='*80)
            
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {args.batch_file}")
        except Exception as e:
            print(f"❌ 배치 처리 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # CSV에서 랜덤 샘플링 (기존 방식)
    else:
        print("=" * 80)
        print("🚀 CSV 파일에서 랜덤 샘플링 DAG 추출")
        print("=" * 80)
        print(f"추출할 메시지 수: {args.num_msgs}개")
        print(f"LLM 모델: {args.llm_model}")
        print(f"프롬프트 모드: {args.prompt_mode}")
        print("=" * 80 + "\n")
        
        dag_finder(num_msgs=args.num_msgs, llm_model_nm=args.llm_model, save_dag_image=args.save_dag_image, prompt_mode=args.prompt_mode)