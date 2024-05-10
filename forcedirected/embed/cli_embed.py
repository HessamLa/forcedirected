import click
import os
import functools
from recursivenamespace import rns

@click.group()
def cli_embed():
    """Main entry point for the graph generation CLI tool."""
    pass

def common_options(func):
    """Decorator to apply common options to graph generation commands."""
    @cli_embed.command(context_settings=dict(show_default=True))
    @click.option('-d', '--n-dim', '--ndim', type=int, default=128, help='Number of dimensions.')
    @click.option('--epochs', type=int, default=1000, help='Number of epochs.', show_default=True)
    @click.option('--lr', type=float, default=1.0, help='Learning rate.', show_default=True)
    @click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda']), default='auto', help='Device to use for computation.', show_default=True)
    @click.option('-n', '--name', type=str, help='Name of the graph.', required=True)
    @click.option('-e', '--edgelist', type=click.Path(), help='Path to the edge list file. Either this or adjlist must be provided. (Somewhat required)')
    @click.option('--adjlist', type=click.Path(), help='Path to the adjacency list. Either this or edgelist must be provided.')
    @click.option('--outdir', type=click.Path(), default='./data', help='Output directory for the embeddings.', show_default=True)
    @click.option('--format',  type=click.Choice(['csv', 'pkl']), default='csv', help='Output file type. csv of Pandas pickle.', show_default=True)
    @click.option('--filename', type=click.Path(), help='Output filename for the embeddings. [default: <dataset>-<method>-d<dim>.<format>]', show_default=False)

    @click.option('--verbosity', type=click.INT, default=2, show_default=True,
                help='Verbosity level as defined in ForceDirected base model. '
                    '0: no output, 1: essential msg, 2: short msg, 3: full msg + exception msg, '
                    '4: full msg + exception msg + raise exception.')
    
    @click.option('--seed', type=int, default=None, help='Random seed for reproducibility.', show_default=True)
    @functools.wraps(func)
    def wrapper(*args, **options):
        options = rns(options)
        if(options.edgelist is None and options.adjlist is None):
            raise ValueError("Either edgelist or adjlist must be provided.")
        return func(*args, **options)
    return wrapper

def save_embeddings(embeddings_df, filepath, format='csv', set_column_names=False):
    """Save the embeddings to a file."""
    if(set_column_names is True):
        # Apply the columns name [id, dim_1, ... dim_d]."""
        columns = [f'dim_{i}' for i in range(embeddings_df.shape[1])]
        columns[0] = 'id'
        embeddings_df.columns = columns
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if(format=='csv'):
        embeddings_df.to_csv(filepath, index=False)
    elif(format=='pkl'):
        embeddings_df.to_pickle(filepath)
    print(f"Embeddings saved to {filepath}")

@common_options
# @click.option('--model', type=str, default='model_201_basic', help='Path to the embeddings file.', show_default=True)
@click.option('--k1', type=float, default=0.999, help='k1 parameter.')
@click.option('--k2', type=float, default=1.0,   help='k2 parameter.')
@click.option('--k3', type=float, default=10.0,  help='k3 parameter.')
@click.option('--k4', type=float, default=0.01,  help='k4 parameter.')
@click.option('--coeffs', nargs=4, type=float, help='Coefficients for the force calculation. If provided, overrides k1, k2, k3, k4. Used to shorten the syntanx.')
def fdbasic(**options):
    """Generate embeddings using the force-directed basic algorithm."""
    options = rns(options)
    if(options.coeffs is not None):
        options.k1 = float(options.coeffs[0])
        options.k2 = float(options.coeffs[1])
        options.k3 = float(options.coeffs[2])
        options.k4 = float(options.coeffs[3])
    
    # Make the filename
    if(options.filename is None):
        options.filename = f"{options.name}-fdbasic-d{options.n_dim}.{options.format}"
    filepath = os.path.join(options.outdir, options.filename)

    print("fd-landmark command with params:")
    for k,v in options.items():
        print(f"{str(k):<16s}: {v}")
    
    if(len(options.edgelist) != 0):
        print("Input graph path     :", options.edgelist)
    elif(len(options.adjlist) != 0):
        print("Input graph path     :", options.adjlist)    
    print("Embedding dimensions :", options.n_dim)
    print("Output directory     :", options.outdir)
    print("Output filename      :", options.filename)

    from .fd import embed_basic
    embeddings_df = embed_basic(**options)

    save_embeddings(embeddings_df, filepath, options.format, set_column_names=True)
    # End of fd_basic 
    #########

@common_options
@click.option('--k1', type=float, default=0.999, help='k1 parameter.')
@click.option('--k2', type=float, default=1.0,   help='k2 parameter.')
@click.option('--k3', type=float, default=10.0,  help='k3 parameter.')
@click.option('--k4', type=float, default=0.01,  help='k4 parameter.')
@click.option('--coeffs', nargs=4, type=float, help='Coefficients for the force calculation. If provided, overrides k1, k2, k3, k4. Used to shorten the syntanx.')
def fdshell(**options):
    """Generate embeddings using the force-directed basic algorithm with shell averaging."""
    options = rns(options)
    if(options.coeffs is not None):
        options.k1 = float(options.coeffs[0])
        options.k2 = float(options.coeffs[1])
        options.k3 = float(options.coeffs[2])
        options.k4 = float(options.coeffs[3])
    
    # Make the filename
    if(options.filename is None):
        options.filename = f"{options.name}-fdshell-d{options.n_dim}.{options.format}"
    filepath = os.path.join(options.outdir, options.filename)

    print("fd-landmark command with params:")
    for k,v in options.items():
        print(f"{str(k):<16s}: {v}")
    
    if(len(options.edgelist) != 0):
        print("Input graph path     :", options.edgelist)
    elif(len(options.adjlist) != 0):
        print("Input graph path     :", options.adjlist)    
    print("Embedding dimensions :", options.n_dim)
    print("Output directory     :", options.outdir)
    print("Output filename      :", options.filename)

    from .fd import embed_shell
    embeddings_df = embed_shell(**options)

    save_embeddings(embeddings_df, filepath, options.format, set_column_names=True)
    # End of fd_shell 
    #########

@common_options
@click.option('-k1','--k1',      type=float, default=0.999, help='k1 parameter.')
@click.option('-k2','--k2',      type=float, default=1.0,   help='k2 parameter.')
@click.option('-k3','--k3',      type=float, default=10.0,  help='k3 parameter.')
@click.option('-k4','--k4',      type=float, default=0.01,  help='k4 parameter.')
@click.option('-a', '--reach_a', type=int,   default=1,    help='Attractive reach parameter.')
@click.option('-r', '--reach_r', type=int,   default=4,    help='Repulsive reach parameter.')
@click.option('-L', '--landmarks_ratio', type=click.FloatRange(0, 1.0), default=0.01, help='Ratio of landmark nodes (top ratio of degrees).')
@click.option('--coeffs', nargs=7, type=(float, float, float, float, int, int, click.FloatRange(0, 1.0)), 
              help='Coefficients for the force calculation. If provided, overrides other parameters. Used to shorten the syntanx.')
def fdtargets(**options):
    """Generate embeddings using the force-directed algorithm with selective target nodes."""
    options = rns(options)
    if(options.coeffs is not None):
        options.k1 = options.coeffs[0]
        options.k2 = options.coeffs[1]
        options.k3 = options.coeffs[2]
        options.k4 = options.coeffs[3]
        options.reach_a = options.coeffs[4]
        options.reach_r = options.coeffs[5]
        options.landmarks_ratio = options.coeffs[6]

    if(options.filename is None):
        options.filename = f"{options.name}-fdtargets-d{options.n_dim}.{options.format}"
    filepath = os.path.join(options.outdir, options.filename)

    print("fd-landmark command with params:")
    for k,v in options.items():
        print(f"{str(k):<16s}: {v}")
    
    if(len(options.edgelist) != 0):
        print("Input graph path     :", options.edgelist)
    elif(len(options.adjlist) != 0):
        print("Input graph path     :", options.adjlist)    
    print("Embedding dimensions :", options.n_dim)
    print("Output directory     :", options.outdir)
    print("Output filename      :", options.filename)

    from .fd import embed_targets
    embeddings_df = embed_targets(**options)

    save_embeddings(embeddings_df, filepath, options.format, set_column_names=True)
    # End of fd_targets
    #########

# node2vec
@common_options
@click.option('-p', '--p', type=float, default=1.0, help='Return parameter.')
@click.option('-q', '--q', type=float, default=1.0, help='In-out parameter.')
@click.option('--walk-length', type=int, default=80, help='Walk length.')
@click.option('--num-walks', type=int, default=20, help='Number of walks.')
@click.option('--context-size', type=int, default=10, help='Window size.')
@click.option('--walks-per-node', type=int, default=10, help='Number of walks per node.')
@click.option('--num-negative-samples', type=int, default=1, help='Number of negative samples.')
@click.option('--epochs', type=int, default=100, help='Number of epochs.', show_default=True) # override the common options
def node2vec(**options):
    """Generate embeddings using the node2vec algorithm."""
    options = rns(options)

    if(options.filename is None):
        options.filename = f"{options.name}-node2vec-d{options.n_dim}.{options.format}"
    filepath = os.path.join(options.outdir, options.filename)

    print("node2vec command with params:")
    for k,v in options.items():
        print(f"{str(k):<16s}: {v}")

    if(len(options.edgelist) != 0):
        print("Input graph path     :", options.edgelist)
    elif(len(options.adjlist) != 0):
        print("Input graph path     :", options.adjlist)    
    print("Embedding dimensions :", options.n_dim)
    print("Output directory     :", options.outdir)
    print("Output filename      :", options.filename)

    from .node2vec import embed_node2vec
    embeddings_df = embed_node2vec(**options)

    save_embeddings(embeddings_df, filepath, options.format, set_column_names=True)
    # End of node2vec
    #########

# graphsage
@common_options
# @click.option('--p', type=float, default=1.0, help='Return parameter.')
# @click.option('--q', type=float, default=0.5, help='In-out parameter.')
# @click.option('--walk-length', type=int, default=80, help='Walk length.')
# @click.option('--num-walks', type=int, default=10, help='Number of walks.')
# @click.option('--context-size', type=int, default=10, help='Window size.')
# @click.option('--walks-per-node', type=int, default=10, help='Number of walks per node.')
# @click.option('--num-negative-samples', type=int, default=1, help='Number of negative samples.')
@click.option('--epochs', type=int, default=100, help='Number of epochs.', show_default=True) # override the common options
def graphsage(**options):
    """Generate embeddings using the GraphSAGE algorithm."""
    options = rns(options)

    if(options.filename is None):
        options.filename = f"{options.name}-graphsage-d{options.n_dim}.{options.format}"
    filepath = os.path.join(options.outdir, options.filename)

    print("graphsage command with params:")
    for k,v in options.items():
        print(f"{str(k):<16s}: {v}")

    if(len(options.edgelist) != 0):
        print("Input graph path     :", options.edgelist)
    elif(len(options.adjlist) != 0):
        print("Input graph path     :", options.adjlist)    
    print("Embedding dimensions :", options.n_dim)
    print("Output directory     :", options.outdir)
    print("Output filename      :", options.filename)

    from .graphsage import embed_graphsage
    embeddings_df = embed_graphsage(**options)

    save_embeddings(embeddings_df, filepath, options.format, set_column_names=True)
    # End of node2vec
    #########

# graphsage
@common_options
# @click.option('--p', type=float, default=1.0, help='Return parameter.')
# @click.option('--q', type=float, default=0.5, help='In-out parameter.')
# @click.option('--walk-length', type=int, default=80, help='Walk length.')
# @click.option('--num-walks', type=int, default=10, help='Number of walks.')
# @click.option('--context-size', type=int, default=10, help='Window size.')
# @click.option('--walks-per-node', type=int, default=10, help='Number of walks per node.')
# @click.option('--num-negative-samples', type=int, default=1, help='Number of negative samples.')
@click.option('--epochs', type=int, default=100, help='Number of epochs.', show_default=True) # override the common options
def line(**options):
    """Generate embeddings using the LINE algorithm."""
    options = rns(options)

    if(options.filename is None):
        options.filename = f"{options.name}-line-d{options.n_dim}.{options.format}"
    filepath = os.path.join(options.outdir, options.filename)

    print("line command with params:")
    for k,v in options.items():
        print(f"{str(k):<16s}: {v}")

    if(len(options.edgelist) != 0):
        print("Input graph path     :", options.edgelist)
    elif(len(options.adjlist) != 0):
        print("Input graph path     :", options.adjlist)    
    print("Embedding dimensions :", options.n_dim)
    print("Output directory     :", options.outdir)
    print("Output filename      :", options.filename)

    from .line import embed_line
    embeddings_df = embed_line(**options)

    save_embeddings(embeddings_df, filepath, options.format, set_column_names=True)
    # End of node2vec
    #########