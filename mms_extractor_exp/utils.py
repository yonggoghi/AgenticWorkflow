import networkx as nx
import graphviz
from graphviz import Digraph
import textwrap
import hashlib
import logging

# 로거 설정
logger = logging.getLogger(__name__)

def create_dag_diagram(G, filename='dag_diagram', wrap_method='record', output_dir=None, **kwargs):
    """
    DAG 시각화 다이어그램 생성 함수
    
    NetworkX 그래프를 Graphviz를 사용하여 시각적 다이어그램으로 변환합니다.
    생성된 이미지는 지정된 디렉토리에 PNG 형태로 저장됩니다.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        시각화할 NetworkX 방향 그래프 객체
    filename : str
        저장할 파일명 (확장자 제외, 기본값: 'dag_diagram')
    wrap_method : str
        텍스트 래핑 방법 ('html_table', 'record', 'manual_wrap', 'fixedsize_false')
    output_dir : str, optional
        출력 디렉토리 경로. None이면 설정에서 자동 선택 (default: None)
    **kwargs : dict
        Graphviz 스타일링 파라미터 (색상, 폰트, 레이아웃 등)
        
    Returns:
    --------
    str or None : 생성된 이미지 파일 경로 (실패 시 None)
    
    Features:
    ---------
    - 연결된 노드만 표시 (고립된 노드 제외)
    - 자동 텍스트 래핑으로 가독성 향상
    - 노드와 엣지의 시각적 구분
    - PNG 형식으로 고품질 이미지 생성
    - 로컬/NAS 저장 위치 선택 가능
    """
    
    # 출력 디렉토리 결정
    if output_dir is None:
        try:
            from config.settings import STORAGE_CONFIG
            output_dir = f'./{STORAGE_CONFIG.get_dag_images_dir()}'
            logger.info(f"📁 저장 위치: {output_dir} ({STORAGE_CONFIG.dag_storage_mode} 모드)")
        except:
            output_dir = './dag_images'  # 기본값
            logger.warning(f"⚠️ 설정 로드 실패, 기본 경로 사용: {output_dir}")
    
    logger.info(f"🎨 DAG 다이어그램 생성 시작 - 파일명: {filename}")
    logger.info(f"📊 입력 그래프 - 노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")
    
    # Step 1: 연결된 노드만 필터링
    # 고립된 노드(엣지가 없는 노드)는 시각화에서 제외
    connected_nodes = set()
    for edge in G.edges():
        connected_nodes.add(edge[0])  # 소스 노드
        connected_nodes.add(edge[1])  # 타겟 노드
    
    if not connected_nodes:
        logger.warning("❌ 그래프에서 연결된 경로를 찾을 수 없습니다")
        print("❌ No connected paths found in the graph")
        return None
    
    # 연결된 노드만으로 서브그래프 생성
    G_connected = G.subgraph(connected_nodes).copy()
    
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
    
    # Process nodes if not path_only
    for node in G_connected.nodes(data=True):
        node_id = str(node[0]).replace(':', '_')
        
        # Determine colors
        in_degree = G_connected.in_degree(node[0])
        out_degree = G_connected.out_degree(node[0])
        
        if in_degree == 0:
            fillcolor, color = '#90EE90', '#228B22'
        elif out_degree == 0:
            fillcolor, color = '#FFB6C1', '#DC143C'
        else:
            fillcolor, color = '#87CEEB', '#4682B4'
        
        # Apply different wrapping methods
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
    try:
        # 출력 디렉토리가 없으면 생성
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("🖼️ DAG 이미지 렌더링 중...")
        output_path = dot.render(filename, directory=output_dir, cleanup=True)
        logger.info(f"✅ DAG 다이어그램 생성 완료: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ DAG 렌더링 중 오류 발생: {e}")
        print(f"❌ Error rendering: {e}")
        return None

def format_node_label(text, wrap_method):
    """
    Format node label based on wrapping method
    
    Returns:
    --------
    tuple: (label, node_attributes_dict)
    """
    
    if wrap_method == 'html_table':
        # Method 1: HTML Table (Best for auto-wrapping)
        # Automatically wraps text to fit table width
        clean_text = text.replace('_', '_ ')  # Add space after colon
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
        clean_text = text.replace('_', '|')  # Record separator
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
                # Wrap each part if too long
                part1 = parts[0]
                part2 = parts[1]
                
                if len(part1) > 10:
                    part1 = '\\n'.join(textwrap.wrap(part1, width=10))
                if len(part2) > 10:
                    part2 = '\\n'.join(textwrap.wrap(part2, width=10))
                
                label = f'{part1}:\\n{part2}'
        else:
            # Wrap long text
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
        # Method 4: Let Graphviz auto-size (simplest)
        label = text.replace(':', ': ')  # Add space for better breaking
        
        node_attrs = {
            'shape': 'box',
            'style': 'rounded,filled',
            'fixedsize': 'false',  # Key: let Graphviz determine size
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

def sha256_hash(text):
    """
    텍스트의 SHA256 해시값 생성
    
    DAG 이미지 파일명 생성에 사용되며, 동일한 메시지는 
    항상 같은 파일명을 가지도록 보장합니다.
    
    Args:
        text (str): 해시할 텍스트 (일반적으로 MMS 메시지)
        
    Returns:
        str: 64자리 16진수 해시값 (안전하고 널리 사용됨)
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()