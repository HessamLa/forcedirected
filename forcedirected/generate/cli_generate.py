import click
@click.group()
def cli_generate():
    """Main entry point for synthetic graph generation CLI tool."""
    pass

import os, sys
import functools
import networkx as nx
from recursivenamespace import rns

@cli_generate.group()
def generate():
    """Generate synthetic graphs using various algorithms."""
    pass

def common_options(func):
    """Decorator to apply common options to graph generation commands."""
    @cli_generate.command(context_settings=dict(show_default=True))
    @click.option('-n', '--n-nodes', type=int, required=True, help='Number of nodes in the graph.')
    @click.option('--basename', type=click.STRING, help='Output basename as a string. The file format is ".edgelist" or ".label". [default: <method>-n<n_nodes>-m<mu>.<format>]')
    @click.option('--outdir', type=click.STRING, default='./data', help='Output directory for the graph file.')    
    @click.option('--seed', type=int, default=None, help='Random seed for reproducibility.')
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if(not os.path.exists(kwargs['outdir'])):
            print("Generating output directory", kwargs['outdir'])
            os.makedirs(kwargs['outdir'], exist_ok=True)
        return func(*args, **kwargs)
    return wrapper # End of common_options(.)


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
    
@common_options
@click.option('--mu', type=float, default=0.5, help='Mixing parameter, fraction of intra-community edges to total edges. (0 <= mu <= 1)')
@click.option('--gamma', type=float, default=2.01, help='Degree power-law distribution. (tau1)')
@click.option('--beta', type=float, default=1.051, help='Community size power-law distribution. (tau2)')
@click.option('--max-community', type=int, default=100, help='Maximum community size.')
@click.option('--min-community', type=int, default=10, help='Minimum community size.')
@click.option('--max-degree', type=int, default=80, help='Maximum degree of nodes.')
@click.option('--average-degree', type=int, default=20, help='Average degree of nodes.')
@click.option('--min-degree', type=int, default=None, help='Minimum degree of nodes (mutually exclusive with average-degree).')
def lfr(**options):
    """Generate a synthetic graph using the LFR benchmark model."""
    from . import lfr as generate_lfr    
    options = rns(options)

    if(options.min_degree is not None): # exclusive with average-degree
        options.average_degree = None
    # generate the graph
    G = generate_lfr(**options)

    # verify
    n = options.n_nodes
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    assert n_nodes == n, f"Expected {n} nodes, got {n_nodes}"
    print("Number of nodes:", n_nodes)
    print("Number of edges:", n_edges)
    
    n_components = nx.number_connected_components(G)
    print("Number of connected components:", n_components) 
    
    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    print("Number of communities:", len(communities))
    
    modularity = nx.community.modularity(G, [set(v) for v in communities])
    print("Modularity:", modularity)
    
    diameter = 0
    if(n_components == 1):
        diameter = nx.diameter(G)
    elif(n_components > 1):
        for c in nx.connected_components(G):
            Gc = G.subgraph(c)
            d = nx.diameter(Gc)
            if(diameter < d):
                diameter = d
    print("Diameter (of the largest component):", diameter)   

    # save to file
    mu = options.mu
    basename = options.basename
    if(basename is None):
        basename = f'lfr-n{n}-m{mu}'
    outpath = os.path.join(options.outdir, basename+'.edgelist')
    click.echo(f"Writing LFR graph edgelist to {outpath}... ")
    nx.write_edgelist(G, outpath, data=False)

    # set the community as a node attribute with key 'label'
    for i, comm in enumerate(communities):
        # i is the community id
        for u in comm:
            G.nodes[u]["label"] = i
    outpath = os.path.join(options.outdir, basename+'.community')
    click.echo(f"Writing LFR graph node labels to {outpath}... ")
    write_labels(G, outpath, attrib="label")
    click.echo("                                          done")
    return # End of lfr(.)


@common_options
@click.option('-c', '--n_communities', type=int, default=5, help='Number of communities.')
@click.option('--p_intra', type=float, default=0.25, help='Number of communities.')
@click.option('--p_inter', type=float, default=0.01, help='Number of communities.')
def sbm(**options):
    """Generate a synthetic graph using the Stochastic Block Model."""
    from . import sbm as generate_sbm
    options = rns(options)
    print("SMB with parameters:")
    for k,v in options.items():
        print(f"{str(k):<16s}: {v}")

    # generate the graph
    Gx = generate_sbm(**options)

    # verify
    n = options.n_nodes
    n_nodes = Gx.number_of_nodes()
    n_edges = Gx.number_of_edges()
    assert n_nodes == n, f"Expected {n} nodes, got {n_nodes}"
    print("Number of nodes:", n_nodes)
    print("Number of edges:", n_edges)
    
    n_components = nx.number_connected_components(Gx)
    print("Number of connected components:", n_components) 
    
    # get the number of blocks in the SBM graph
    blocks = {}
    for v in Gx:
        block_id = Gx.nodes[v]["block"]
        if(block_id not in blocks):
            blocks[block_id] = [v]
        else:
            blocks[block_id].append(v)
    print("Number of blocks:", len(blocks))

    # Get the modularity
    modularity = nx.community.modularity(Gx, [set(v) for v in blocks.values()])
    print("Modularity:", modularity)

    # communities = frozenset((G.nodes[v]["community"] for v in G))
    # assert len(communities) == options.n_communities
    # print("Number of communities:", len(communities))
    
    diameter = 0
    if(n_components == 1):
        diameter = nx.diameter(Gx)
    elif(n_components > 1):
        for c in nx.connected_components(Gx):
            Gc = Gx.subgraph(c)
            d = nx.diameter(Gc)
            if(diameter < d):
                diameter = d
    print("Diameter (of the largest component):", diameter)   

    # save to file
    n_comm = options.n_communities
    p1 = options.p_intra
    p2 = options.p_inter
    basename = options.basename
    if(basename is None):
        basename = f'sbm-n{n}-c{n_comm}-p1{p1}-p2{p2}'
    outpath = os.path.join(options.outdir, basename+'.edgelist')
    click.echo(f"Writing SBM graph to {outpath} with paramters n={n}, communitites={n_comm}, p-intra={p1}, p-inter={p2}.")
    nx.write_edgelist(Gx, outpath, data=False)

    outpath = os.path.join(options.outdir, basename+'.community')
    click.echo(f"Writing SBM graph node labels to {outpath}... ")
    write_labels(Gx, outpath, attrib="block")
    click.echo("                                          done")
    return # End of lfr(.)

@common_options
@click.option('--param1', type=float, help='Parameter 1 specific to binary tree.')
@click.option('--param2', type=int, help='Parameter 2 specific to binary tree.')
@click.option('--outpath', type=click.Path(), default='./bintree.edgelist', help='Output path for the graph file.')
def binary_tree(**options):
    # Implement binary tree generation logic here
    click.echo(f"Binary tree generated at {options['outpath']} with node count {options['n']} and parameters param1={options['param1']}, param2={options['param2']}.")
    return # End of binary_tree(.)


