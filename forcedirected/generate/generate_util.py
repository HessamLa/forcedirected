import networkx as nx

def write_edgelist(Gx, filepath, attrib=None):
    """
    Writes the edge list of a NetworkX graph to a file, including specified edge attributes.
    
    Args:
    - Gx: NetworkX graph object.
    - filepath: The path to the file where the edge list will be saved.
    - attrib: Optional; the name of the edge attribute to include in the file.
    """
    if attrib:
        # Write edges with attributes
        with open(filepath, 'w') as file:
            for u, v, data in Gx.edges(data=True):
                if attrib in data:
                    file.write(f"{u} {v} {data[attrib]}\n")
                else:
                    file.write(f"{u} {v}\n")
    else:
        # Write edges without attributes
        nx.write_edgelist(Gx, filepath, data=False)

def write_labels(Gx, filepath, attrib=None):
    """
    Writes the nodes of a NetworkX graph to a file, including specified node attributes.
    
    Args:
    - Gx: NetworkX graph object.
    - filepath: The path to the file where the nodes will be saved.
    - attrib: Optional; the name of the node attribute to include in the file.
    """
    # return
    with open(filepath, 'w') as file:
        for u in Gx.nodes():
            try:
                file.write(f"{u} {Gx.nodes[u][attrib]}\n")
            except KeyError:
                file.write(f"{u}\n")