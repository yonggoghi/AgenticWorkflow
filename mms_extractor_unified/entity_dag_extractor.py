from concurrent.futures import ThreadPoolExecutor
import time
import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
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

stop_item_names = pd.read_csv(settings.METADATA_CONFIG.stop_items_path)['stop_words'].to_list()
mms_pdf = pd.read_csv(settings.METADATA_CONFIG.mms_msg_path)
mms_pdf = mms_pdf.astype('str')

###############################################################################
# 1) 기존 정규식 및 파서 (하위 호환성 유지)
###############################################################################
PAT = re.compile(r"\((.*?)\)\s*-\[(.*?)\]->\s*\((.*?)\)")
NODE_ONLY = re.compile(r"\((.*?)\)\s*$")

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
# 2) 개선된 노드 스플리터 – 유연한 파트 처리
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
# 3) 기존 DAG 빌더 (하위 호환성 유지)
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
    
    return g

###############################################################################
# 4) 기존 Path Finder (하위 호환성 유지)
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


def extract_dag(parser:DAGParser, msg: str, llm_model):
    """
    DAG 추출 메인 함수
    
    메시지에서 엔티티 간의 관계를 DAG(Directed Acyclic Graph) 형태로 추출합니다.
    
    Args:
        parser (DAGParser): DAG 파싱을 위한 파서 객체
        msg (str): 분석할 MMS 메시지 텍스트
        llm_model: 사용할 LLM 모델 (Langchain 호환)
        
    Returns:
        dict: {
            'dag_section': str,     # 파싱된 DAG 텍스트 표현
            'dag': nx.DiGraph,      # NetworkX 그래프 객체  
            'dag_raw': str          # LLM 원본 응답
        }
        
    Process:
        1. LLM을 통해 엔티티 관계 추출
        2. DAG 섹션 파싱
        3. NetworkX 그래프 구조 생성
        4. 결과 반환
    """
    
    logger.info("🚀 DAG 추출 프로세스 시작")
    logger.info(f"📝 입력 메시지 길이: {len(msg)}자")
    logger.info(f"🤖 사용 LLM 모델: {llm_model}")
    
    prompt = f"""
## 작업 목표
통신사 광고 메시지에서 **핵심 행동 흐름**을 추출하여 간결한 DAG(Directed Acyclic Graph) 형식으로 표현

## 핵심 원칙
1. **최소 경로 원칙**: 동일한 결과를 얻는 가장 짧은 경로만 표현
2. **핵심 흐름 우선**: 부가적 설명보다 주요 행동 연쇄에 집중
3. **중복 제거**: 의미가 겹치는 노드는 하나로 통합

## 출력 형식
```
(개체명:기대행동) -[관계동사]-> (개체명:기대행동)
또는
(개체명:기대행동)
```

## 개체명 카테고리
### 필수 추출 대상
- **제품/서비스**: 구체적 제품명, 서비스명, 요금제명
- **핵심 혜택**: 금전적 혜택, 사은품, 할인
- **행동 장소**: 온/오프라인 채널 (필요 시만)

### 추출 제외 대상
- 광고 대상자 (예: "아이폰 고객님")
- 일정/기간 정보
- 부가 설명 기능 (핵심 흐름과 무관한 경우)
- 법적 고지사항

## 기대 행동 (10개 표준 동사)
`구매, 가입, 사용, 방문, 참여, 등록, 다운로드, 확인, 수령, 적립`

## 관계 동사 우선순위
### 1순위: 조건부 관계
- `가입하면`, `구매하면`, `사용하면`
- `가입후`, `구매후`, `사용후`

### 2순위: 결과 관계
- `받다`, `수령하다`, `적립하다`

### 3순위: 경로 관계 (필요 시만)
- `통해`, `이용하여`

## DAG 구성 전략
### Step 1: 모든 가치 제안 식별
광고에서 제시하는 **모든 독립적 가치**를 파악
- **즉각적 혜택**: 금전적 보상, 사은품 등
- **서비스 가치**: 제품/서비스 자체의 기능과 혜택
예: "네이버페이 5000원" + "AI 통화 기능 무료 이용"

### Step 2: 독립 경로 구성
각 가치 제안별로 별도 경로 생성:
1. **혜택 획득 경로**: 가입 → 사용 → 보상
2. **서비스 체험 경로**: 가입 → 경험 → 기능 활용

### Step 3: 세부 기능 표현
주요 기능들이 명시된 경우 분기 구조로 표현:
- 통합 가능한 기능은 하나로 (예: AI통화녹음/요약 → "AI통화기능")
- 독립적 기능은 별도로 (예: AI스팸필터링)

## 분석 프로세스 (필수 단계)
### Step 1: 메시지 이해
- 전체 메시지를 한 문단으로 요약
- 광고주의 의도 파악
- **암시된 행동 식별**: 명시되지 않았지만 필수적인 행동 (예: 매장 방문)

### Step 2: 가치 제안 식별
- 즉각적 혜택 (금전, 사은품 등)
- 서비스 가치 (기능, 편의성 등)
- 부가 혜택 (있다면)

### Step 3: Root Node 결정
- **사용자의 첫 번째 행동은 무엇인가?**
- 매장 주소/연락처가 있다면 → 방문이 시작점
- 온라인 링크가 있다면 → 접속이 시작점
- 앱 관련 내용이라면 → 다운로드가 시작점

### Step 4: 관계 분석
- Root Node부터 시작하는 전체 흐름
- 각 행동 간 인과관계 검증
- 조건부 관계 명확화
- 시간적 순서 확인

### Step 5: DAG 구성
- 위 분석을 바탕으로 노드와 엣지 결정
- 중복 제거 및 통합

### Step 6: 자기 검증 및 수정
- **평가**: 초기 DAG를 아래 기준에 따라 검토
  - Root Node가 명확히 식별되었는가? (방문/접속/다운로드 등)
  - 모든 독립적 가치 제안이 포함되었는가? (즉각적 혜택과 서비스 가치)
  - 주요 기능들이 적절히 그룹화되었는가? (중복 제거 여부)
  - 각 경로가 명확한 가치를 전달하는가?
  - 전체 구조가 간결하고 이해하기 쉬운가?
  - 관계 동사가 우선순위에 맞게 사용되었는가? (조건부 > 결과 > 경로)
- **문제 식별**: 위 기준 중 충족되지 않은 항목을 명시하고, 그 이유를 설명
- **수정**: 식별된 문제를 해결한 수정된 DAG를 생성

### Step 7: 최종 검증
- 수정된 DAG가 모든 기준을 충족하는지 재확인
- 만약 문제가 남아있다면, 추가 수정 수행 (최대 2회 반복)
- 최종적으로 모든 기준이 충족되었음을 확인

## 출력 형식 (반드시 모든 섹션 포함)
### 1. 메시지 분석
```
[메시지 요약 및 핵심 의도]
[식별된 가치 제안 목록]
```

### 2. 초기 DAG
```
[초기 DAG 구조]
```

### 3. 자기 검증 결과
```
[평가 기준별 검토 결과]
[식별된 문제점 및 이유]
```

### 4. 수정된 DAG
```
[수정된 DAG 구조]
```

### 5. 최종 검증 및 추출 근거
```
[최종 DAG가 모든 기준을 충족하는 이유]
[노드/엣지 선택의 논리적 근거]
```

## 실행 지침
1. 위 7단계 분석 프로세스를 **순서대로** 수행
2. 각 단계에서 발견한 내용을 **명시적으로** 기록
3. DAG 구성 전 **충분한 분석** 수행
4. 초기 DAG 생성 후 **반드시 자기 검증 및 수정** 수행
5. 최종 출력에 **모든 섹션** 포함
6. **중요**: 분석 과정을 생략하지 말고, 사고 과정과 수정 이유를 투명하게 보여주세요

## 예시 분석
### 잘못된 예시 (핵심 흐름만 추출)
```
(에이닷:가입) -[가입후]-> (AI전화서비스:사용)
(AI전화서비스:사용) -[사용하면]-> (네이버페이5000원:수령)
```
→ 문제: 서비스 자체의 가치(AI 기능들)가 누락됨

### 올바른 예시 (완전한 가치 표현 - Root Node 포함)
```
# 매장 방문부터 시작하는 경로
(제이스대리점:방문) -[방문하여]-> (갤럭시S21:구매)
(갤럭시S21:구매) -[구매시]-> (5GX프라임요금제:가입)
(5GX프라임요금제:가입) -[가입하면]-> (지원금45만원+15%:수령)

# 온라인 시작 경로 예시
(T다이렉트샵:접속) -[접속하여]-> (갤럭시S24:구매)
(갤럭시S24:구매) -[구매시]-> (사은품:수령)
```
→ 장점: 사용자의 첫 행동(Root Node)부터 명확히 표현

## message:
{msg}
"""
    
    logger.info("🤖 LLM에 DAG 추출 요청 중...")
    logger.info(f"📏 프롬프트 길이: {len(prompt)}자")
    
    # Step 1: LLM을 통한 엔티티 관계 추출
    try:
        dag_raw = llm_model.invoke(prompt).content
        logger.info(f"📝 LLM 응답 길이: {len(dag_raw)}자")
        logger.info(f"📄 LLM 응답 미리보기: {dag_raw[:200]}...")
    except Exception as e:
        logger.error(f"❌ LLM 호출 중 오류 발생: {e}")
        raise

    # Step 2: DAG 섹션 추출 및 정리
    # LLM 응답에서 실제 DAG 구조 부분만 추출
    logger.info("🔍 DAG 섹션 추출 중...")
    dag_section = parser.extract_dag_section(dag_raw)
    logger.info(f"📄 추출된 DAG 섹션 길이: {len(dag_section)}자")
    
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
def dag_finder(num_msgs=50, llm_model_nm='ax', save_dag_image=True):

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
    output_file = "./logs/dag_extraction_output.txt"

    line_break_patterns = {"__":"\n", "■":"\n■", "▶":"\n▶", "_":"\n"}
    
    # 개선된 파서 초기화
    parser = DAGParser() 
    with open(output_file, 'a', encoding='utf-8') as f:
        # 실행 시작 시점 기록
        from datetime import datetime
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'='*80}\n")
        f.write(f"DAG 추출 실행 시작: {start_time}\n")
        f.write(f"{'='*80}\n\n")
        
        for msg in random.sample(mms_pdf.query("msg.str.contains('')")['msg'].unique().tolist(), num_msgs):
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
                extract_dag_result = extract_dag(parser, msg, llm_model)

                dag_raw = extract_dag_result['dag_raw']
                dag_section = extract_dag_result['dag_section']
                dag = extract_dag_result['dag']

                print(dag_raw)
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
                        nodes, edges = parse_block(re.sub(r'^```|```$', '', dag_raw.strip()))
                        dag = build_dag(nodes, edges)
                else:
                    # 기본 파서 사용
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

    if save_dag_image:
        create_dag_diagram(dag, filename=f'dag_{sha256_hash(msg)}')
        print(f"DAG 이미지가 저장되었습니다: {f'dag_{sha256_hash(msg)}.png'}")

if __name__ == "__main__":
    import argparse
    
    parser_arg = argparse.ArgumentParser(description='DAG 추출기')
    parser_arg.add_argument('--num_msgs', type=int, default=50, help='추출할 메시지 수')
    parser_arg.add_argument('--llm_model', type=str, default='ax', help='사용할 LLM 모델')
    parser_arg.add_argument('--save_dag_image', type=bool, default=True, help='DAG 이미지 저장 여부')
    args = parser_arg.parse_args()
    dag_finder(num_msgs=args.num_msgs, llm_model_nm=args.llm_model, save_dag_image=args.save_dag_image)