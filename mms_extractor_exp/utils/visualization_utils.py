import os
import logging
import textwrap
import networkx as nx
from graphviz import Digraph

logger = logging.getLogger(__name__)

def format_node_label(text, wrap_method='record'):
    """
    Format node label based on wrapping method
    
    Returns:
    --------
    tuple: (label, node_attributes_dict)
    """
    
    if wrap_method == 'html_table':
        # Method 1: HTML Table (Best for auto-wrapping)
        clean_text = text.replace('_', '_ ')
        label = f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">' \
                f'<TR><TD BALIGN="CENTER">{clean_text}</TD></TR></TABLE>>'
        
        node_attrs = {
            'shape': 'box',
            'style': 'rounded,filled',
            'width': '2.0',
            'height': '0.8',
            'fixedsize': 'false',
            'margin': '0.1,0.1'
        }
        
    elif wrap_method == 'record':
        # Method 2: Record shape (Good for structured text)
        clean_text = text.replace('_', '|')
        label = f'{{{clean_text}}}'
        
        node_attrs = {
            'shape': 'record',
            'style': 'rounded,filled',
            'fixedsize': 'false',
            'margin': '0.2,0.1'
        }
        
    elif wrap_method == 'manual_wrap':
        # Method 3: Manual text wrapping
        if ':' in text:
            parts = text.split(':')
            if len(parts) == 2:
                part1 = parts[0]
                part2 = parts[1]
                
                if len(part1) > 10:
                    part1 = '\\n'.join(textwrap.wrap(part1, width=10))
                if len(part2) > 10:
                    part2 = '\\n'.join(textwrap.wrap(part2, width=10))
                
                label = f'{part1}:\\n{part2}'
        else:
            if len(text) > 12:
                wrapped = textwrap.wrap(text, width=12)
                label = '\\n'.join(wrapped)
            else:
                label = text
        
        node_attrs = {
            'shape': 'box',
            'style': 'rounded,filled',
            'fixedsize': 'false',
            'margin': '0.3,0.2'
        }
        
    elif wrap_method == 'fixedsize_false':
        # Method 4: Let Graphviz auto-size
        label = text.replace(':', ': ')
        
        node_attrs = {
            'shape': 'box',
            'style': 'rounded,filled',
            'fixedsize': 'false',
            'margin': '0.3,0.2'
        }
        
    else:  # default
        label = text
        node_attrs = {
            'shape': 'box',
            'style': 'rounded,filled',
            'width': '1.5',
            'height': '0.8',
            'fixedsize': 'true'
        }
    
    return label, node_attrs

def create_dag_diagram(dag: nx.DiGraph, filename: str = "dag", save_dir: str = "dag_images", wrap_method: str = 'record', **kwargs):
    """
    DAG 시각화 다이어그램 생성 함수
    
    NetworkX 그래프를 Graphviz를 사용하여 시각적 다이어그램으로 변환합니다.
    
    Args:
        dag: NetworkX DiGraph 객체
        filename: 저장할 파일명 (확장자 제외)
        save_dir: 저장할 디렉토리
        wrap_method: 텍스트 래핑 방법 ('html_table', 'record', 'manual_wrap', 'fixedsize_false')
        **kwargs: Graphviz 스타일링 파라미터
        
    Returns:
        str or None: 생성된 이미지 파일 경로 (실패 시 None)
    """
    
    logger.info(f"🎨 DAG 다이어그램 생성 시작 - 파일명: {filename}")
    logger.info(f"📊 입력 그래프 - 노드 수: {dag.number_of_nodes()}, 엣지 수: {dag.number_of_edges()}")
    
    # Graphviz 실행 파일 경로 설정 (PATH 문제 해결)
    import os as os_module
    graphviz_path = '/usr/local/bin'
    if graphviz_path not in os_module.environ.get('PATH', ''):
        os_module.environ['PATH'] = f"{graphviz_path}:{os_module.environ.get('PATH', '')}"
        logger.info(f"✅ Graphviz PATH 추가: {graphviz_path}")
    
    try:
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # Step 1: 연결된 노드만 필터링
        connected_nodes = set()
        for edge in dag.edges():
            connected_nodes.add(edge[0])
            connected_nodes.add(edge[1])
        
        if not connected_nodes:
            logger.warning("❌ 그래프에서 연결된 경로를 찾을 수 없습니다")
            return None
        
        # 연결된 노드만으로 서브그래프 생성
        G_connected = dag.subgraph(connected_nodes).copy()
        
        # Step 2: Graphviz 기본 파라미터 설정
        default_params = {
            'engine': 'dot',
            'format': 'png',
            'graph_attr': {
                'rankdir': 'LR',
                'size': '10,4',
                'dpi': '300',
                'bgcolor': 'white',
                'fontname': 'Arial',
                'fontsize': '11',
                'pad': '0.3',
                'ranksep': '1.2',
                'nodesep': '0.8',
                'splines': 'false',
                'concentrate': 'false',
                'ordering': 'out',
                'minlen': '1',
                'overlap': 'false'
            },
            'node_attr': {
                'shape': 'record',
                'style': 'rounded,filled',
                'fontname': 'Arial',
                'fontsize': '11',
                'fontcolor': 'black',
                'penwidth': '2',
                'fixedsize': 'false'
            },
            'edge_attr': {
                'fontname': 'Arial',
                'fontsize': '12',
                'color': 'darkblue',
                'arrowsize': '1.0',
                'arrowhead': 'normal',
                'penwidth': '3',
                'fontcolor': 'darkred',
                'minlen': '1',
                'len': '1.0'
            }
        }
        
        # Update with user parameters
        params = {**default_params, **kwargs}
        
        # Create Graphviz Digraph
        dot = Digraph(name=filename, engine=params['engine'], format=params.get('format', 'png'))
        
        # Set attributes
        for key, value in params['graph_attr'].items():
            dot.graph_attr[key] = str(value)
        for key, value in params['node_attr'].items():
            dot.node_attr[key] = str(value)
        for key, value in params['edge_attr'].items():
            dot.edge_attr[key] = str(value)
        
        # Process nodes
        for node in G_connected.nodes(data=True):
            node_id = str(node[0]).replace(':', '_')
            
            # Determine colors based on node position
            in_degree = G_connected.in_degree(node[0])
            out_degree = G_connected.out_degree(node[0])
            
            if in_degree == 0:
                fillcolor, color = '#90EE90', '#228B22'  # Green for start nodes
            elif out_degree == 0:
                fillcolor, color = '#FFB6C1', '#DC143C'  # Pink for end nodes
            else:
                fillcolor, color = '#87CEEB', '#4682B4'  # Blue for middle nodes
            
            # Apply wrapping method
            label, node_attrs = format_node_label(node_id, wrap_method)
            
            dot.node(node_id, label=label, fillcolor=fillcolor, color=color, **node_attrs)
        
        # Add edges
        for edge in G_connected.edges(data=True):
            source = str(edge[0]).replace(':', '_')
            target = str(edge[1]).replace(':', '_')
            edge_data = edge[2] if len(edge) > 2 else {}
            
            # Edge attributes
            edge_attrs = {}
            if 'label' in edge_data or 'relation' in edge_data:
                edge_attrs['label'] = f' {edge_data["relation"]} ' if 'relation' in edge_data else f' {edge_data["label"]} '
                edge_attrs['fontsize'] = '12'
                edge_attrs['fontcolor'] = 'darkred'
            
            dot.edge(source, target, **edge_attrs)
        
        # Render
        logger.info("🖼️ DAG 이미지 렌더링 중...")
        output_path = dot.render(filename, directory=save_dir, cleanup=True)
        logger.info(f"✅ DAG 다이어그램 생성 완료: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ DAG 다이어그램 생성 실패: {e}")
        return None
