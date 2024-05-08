import click
import os, sys
import functools
import networkx as nx
from recursivenamespace import rns

@click.group()
def cli_generate():
    """Main entry point for the graph generation CLI tool."""
    pass

@cli_generate.group()
def generate():
    """Generate synthetic graphs using various algorithms."""
    pass

def common_options(func):
    """Decorator to apply common options to graph generation commands."""
    @cli_generate.command(context_settings=dict(show_default=True))
    @click.option('-n', '--n-nodes', type=int, required=True, help='Number of nodes in the graph.')
    @click.option('--filename', type=click.STRING, help='Output filename as a string. [default: <method>-n<n_nodes>-m<mu>.<format>]')
    @click.option('--outdir', type=click.STRING, default='./data', help='Output directory for the graph file.')    
    @click.option('--format', type=click.Choice(['edgelist', 'adjlist']), default='edgelist', help='Output path for the graph file.', required=False)
    @click.option('--seed', type=int, default=None, help='Random seed for reproducibility.')
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper # End of common_options(.)


def write_graph(G, filepath, fmt='edgelist', data=False, msg=True, **kwargs):
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("Invalid graph type. Only NetworkX graphs are supported.")
    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    if fmt == 'edgelist':
        nx.write_edgelist(G, filepath, data=data)
    elif fmt == 'adjlist':
        nx.write_adjlist(G, filepath)
    else:
        raise ValueError("Unsupported format")

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
    from . import lfr    
    options = rns(options)
    # generate the graph
    G = lfr.generate(**options)

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
    fmt = options.format
    filename = options.filename
    if(filename is None):
        filename = f'lfr-n{n}-m{mu}.{fmt}'
    outpath = os.path.join(options.outdir, filename)
    click.echo(f"Writing LFR graph to {outpath}... ")
    write_graph(G, outpath, fmt=fmt, data=False, msg="")
    click.echo("                                          done")
    return # End of lfr(.)


@common_options
# @click.option('--mu', type=float, default=0.5, help='Mixing parameter, fraction of intra-community edges to total edges. (0 <= mu <= 1)')
# @click.option('--gamma', type=float, default=2.01, help='Degree power-law distribution. (tau1)')
# @click.option('--beta', type=float, default=1.051, help='Community size power-law distribution. (tau2)')
# @click.option('--max-community', type=int, default=100, help='Maximum community size.')
# @click.option('--min-community', type=int, default=10, help='Minimum community size.')
# @click.option('--max-degree', type=int, default=80, help='Maximum degree of nodes.')
# @click.option('--average-degree', type=int, default=20, help='Average degree of nodes.')
# @click.option('--min-degree', type=int, default=None, help='Minimum degree of nodes (mutually exclusive with average-degree).')
def sbm(**options):
    """Generate a synthetic graph using the Stochastic Block Model."""
    from .sbm import generate_sbm
    options = rns(options)
    # generate the graph
    G = generate_sbm(**options)

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

    # UPDATE THE FOLLOWING CODE
    print("Not implemented yet.")
    exit(1)
    # # save to file
    # mu = options.mu
    fmt = options.format
    # filename = options.filename
    if(filename is None):
        filename = f'lfr-n{n}-m{mu}.{fmt}'
    outpath = os.path.join(options.outdir, filename)
    click.echo(f"Writing LFR graph to {outpath}... ")
    write_graph(G, outpath, fmt=fmt, data=False, msg="")
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


