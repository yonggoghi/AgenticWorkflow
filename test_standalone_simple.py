#!/usr/bin/env python3
"""
독립형 노드 파싱 기능 간단 테스트
"""

import re
from typing import List, Tuple, Union, Dict, Any, Optional, Set
import networkx as nx

class SimpleDAGParser:
    """간단한 DAG 파서 - 독립형 노드 테스트용"""
    
    def __init__(self):
        # 엣지 패턴
        self.edge_pattern = r'\(([^:)]+):([^)]+)\)\s*-\[([^\]]+)\]->\s*\(([^:)]+):([^)]+)\)'
        # 독립형 노드 패턴
        self.standalone_node_pattern = r'\(([^:)]+):([^)]+)\)\s*$'
    
    def parse_dag_line(self, line: str) -> Optional[Union[Tuple[str, str, str, str, str], Tuple[str, str]]]:
        """단일 DAG 라인을 파싱하여 구성 요소 반환"""
        # 먼저 엣지 패턴 확인
        edge_match = re.match(self.edge_pattern, line)
        if edge_match:
            return (
                edge_match.group(1).strip(),  # src_entity
                edge_match.group(2).strip(),  # src_action
                edge_match.group(3).strip(),  # relation
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
    
    def parse_dag(self, dag_text: str) -> nx.DiGraph:
        """DAG 텍스트를 NetworkX DiGraph로 변환"""
        G = nx.DiGraph()
        
        # 통계 정보 저장
        stats = {
            'total_edges': 0,
            'standalone_nodes': 0,
            'parse_errors': [],
            'parsed_lines': []
        }
        
        for line_num, line in enumerate(dag_text.strip().split('\n'), 1):
            line = line.strip()
            
            # 빈 라인이나 주석 라인 건너뛰기
            if not line or line.startswith('#'):
                continue
            
            # DAG 엣지 또는 독립형 노드 파싱
            parsed = self.parse_dag_line(line)
            if parsed:
                try:
                    if len(parsed) == 5:  # 엣지
                        src_entity, src_action, relation, dst_entity, dst_action = parsed
                        
                        # 노드 ID 생성
                        src_node = f"{src_entity}:{src_action}"
                        dst_node = f"{dst_entity}:{dst_action}"
                        
                        # 노드 및 엣지 추가
                        G.add_node(src_node, entity=src_entity, action=src_action)
                        G.add_node(dst_node, entity=dst_entity, action=dst_action)
                        G.add_edge(src_node, dst_node, relation=relation)
                        
                        stats['total_edges'] += 1
                        stats['parsed_lines'].append(f"Line {line_num}: {src_node} -[{relation}]-> {dst_node}")
                        
                    elif len(parsed) == 2:  # 독립형 노드
                        entity, action = parsed
                        
                        # 노드 ID 생성
                        node_id = f"{entity}:{action}"
                        
                        # 독립형 노드 추가
                        G.add_node(node_id, entity=entity, action=action)
                        
                        stats['standalone_nodes'] += 1
                        stats['parsed_lines'].append(f"Line {line_num}: Standalone node {node_id}")
                    
                except Exception as e:
                    stats['parse_errors'].append(f"Line {line_num}: {str(e)}")
            else:
                # 파싱 실패한 라인 기록
                stats['parse_errors'].append(f"Line {line_num}: 패턴 매칭 실패 - {line[:50]}...")
        
        # 그래프에 통계 정보 저장
        G.graph['stats'] = stats
        
        return G

def test_standalone_node_parsing():
    """독립형 노드 파싱 테스트"""
    parser = SimpleDAGParser()
    
    # 테스트 케이스들
    test_cases = [
        # 독립형 노드들
        "(에이닷:가입)",
        "(네이버페이5000원:수령)",
        "(AI전화서비스:사용)",
        
        # 엣지들  
        "(에이닷:가입) -[가입후]-> (AI전화서비스:사용)",
        
        # 혼합 DAG 텍스트
        """(에이닷:가입)
(에이닷:가입) -[가입후]-> (AI전화서비스:사용)
(AI전화서비스:사용) -[사용하면]-> (네이버페이5000원:수령)
(AI스팸필터링:사용)"""
    ]
    
    print("=== 독립형 노드 파싱 테스트 ===")
    
    # 개별 라인 테스트
    for i, test_case in enumerate(test_cases[:4], 1):
        print(f"\n테스트 {i}: {test_case}")
        result = parser.parse_dag_line(test_case)
        if result:
            if len(result) == 2:
                entity, action = result
                print(f"  ✓ 독립형 노드 파싱 성공: entity='{entity}', action='{action}'")
            elif len(result) == 5:
                src_entity, src_action, relation, dst_entity, dst_action = result
                print(f"  ✓ 엣지 파싱 성공: {src_entity}:{src_action} -[{relation}]-> {dst_entity}:{dst_action}")
        else:
            print(f"  ✗ 파싱 실패")
    
    # 혼합 DAG 테스트
    print(f"\n테스트 5: 혼합 DAG 파싱")
    mixed_dag = test_cases[4]
    print(f"입력 텍스트:\n{mixed_dag}")
    
    try:
        G = parser.parse_dag(mixed_dag)
        print(f"\n  ✓ 혼합 DAG 파싱 성공!")
        
        stats = G.graph.get('stats', {})
        print(f"  - 노드 수: {G.number_of_nodes()}")
        print(f"  - 엣지 수: {G.number_of_edges()}")
        print(f"  - 독립형 노드 수: {stats.get('standalone_nodes', 0)}")
        print(f"  - 성공한 라인: {len(stats.get('parsed_lines', []))}")
        print(f"  - 에러 라인: {len(stats.get('parse_errors', []))}")
        
        print(f"\n  노드 목록:")
        for node in G.nodes(data=True):
            print(f"    - {node[0]}: entity='{node[1].get('entity', '')}', action='{node[1].get('action', '')}'")
        
        print(f"\n  엣지 목록:")
        for edge in G.edges(data=True):
            if 'relation' in edge[2]:
                print(f"    - {edge[0]} -[{edge[2]['relation']}]-> {edge[1]}")
        
        if stats.get('parsed_lines'):
            print(f"\n  파싱된 라인들:")
            for line in stats['parsed_lines']:
                print(f"    ✓ {line}")
        
        if stats.get('parse_errors'):
            print(f"\n  파싱 에러들:")
            for error in stats['parse_errors']:
                print(f"    ✗ {error}")
                
    except Exception as e:
        print(f"  ✗ 혼합 DAG 파싱 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_standalone_node_parsing()
