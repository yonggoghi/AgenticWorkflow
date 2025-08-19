import networkx as nx
import graphviz
from graphviz import Digraph
import textwrap
import hashlib

def create_dag_diagram(G, filename='dag_diagram', wrap_method='record', **kwargs):
    """
    Create diagram with automatic text wrapping based on node size
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The graph object to visualize
    filename : str
        Base filename (without extension)
    wrap_method : str
        Text wrapping method: 'html_table', 'record', 'manual_wrap', 'fixedsize_false'
    **kwargs : dict
        Graphviz styling parameters
    """
    
    # Filter connected nodes
    connected_nodes = set()
    for edge in G.edges():
        connected_nodes.add(edge[0])
        connected_nodes.add(edge[1])
    
    if not connected_nodes:
        print("❌ No connected paths found in the graph")
        return None
    
    G_connected = G.subgraph(connected_nodes).copy()
    
    # Default parameters
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
        output_path = dot.render(filename, directory='./dag_images', cleanup=True)
        return output_path
    except Exception as e:
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
    """SHA-256 해시 생성 (안전하고 널리 사용됨)"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()