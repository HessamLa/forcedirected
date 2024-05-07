import os, sys
import networkx as nx

def write_graph(G, filepath, fmt='edgelist', data=False, msg=True, **kwargs):
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("Invalid graph type. Only NetworkX graphs are supported.")

    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    # filename = f'graph.{fmt}'
    # full_path = os.path.join(filepath, filename)

    if fmt == 'edgelist':
        nx.write_edgelist(G, filepath, data=data)
    elif fmt == 'adjlist':
        nx.write_adjlist(G, filepath)
    # elif fmt == 'gml':
    #     try:
    #         nx.write_gml(G, filepath)
    #     except Exception as e:
    #         print(f"Error writing GML file: {e}")
    #         sys.exit(1)
            
    # elif fmt == 'gexf':
    #     nx.write_gexf(G, filepath)
    # elif fmt == 'graphml':
    #     nx.write_graphml(G, filepath)
    else:
        raise ValueError("Unsupported format")
    
