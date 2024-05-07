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
    @click.option('--format', type=click.Choice(['csv', 'pkl']), default='csv', help='Output file type. csv of Pandas pickle.', show_default=True)
    @click.option('--filename', type=click.Path(), help='Output filename for the embeddings. [default: <method>-n<node-count>.<format>]', show_default=False)

    @click.option('--verbosity', type=click.INT, default=2, show_default=True,
                help='Verbosity level as defined in ForceDirected base model. '
                    '0: no output, 1: essential msg, 2: short msg, 3: full msg + exception msg, '
                    '4: full msg + exception msg + raise exception.')
    
    @click.option('--seed', type=int, default=None, help='Random seed for reproducibility.', show_default=True)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs = rns(kwargs)
        if(kwargs.edgelist is None and kwargs.adjlist is None):
            raise ValueError("Either edgelist or adjlist must be provided.")
        if(kwargs.filename is None):
            kwargs.filename = f"{kwargs.name}-d{kwargs.n_dim}.{kwargs.format}"
        return func(*args, **kwargs)
    return wrapper

def save_embeddings(embeddings_df, filepath, format):
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
def fd_basic(**kwargs):
    """Generate embeddings using the force-directed basic algorithm."""
    kwargs = rns(kwargs)
    if(kwargs.coeffs is not None):
        kwargs.k1 = float(kwargs.coeffs[0])
        kwargs.k2 = float(kwargs.coeffs[1])
        kwargs.k3 = float(kwargs.coeffs[2])
        kwargs.k4 = float(kwargs.coeffs[3])
    
    print("fd-basic command with params:")
    for k,v in kwargs.items():
        print(f"{str(k):<16s}: {v}")
    
    # Make the filename
    filepath = os.path.join(kwargs.outdir, kwargs.filename)

    if(len(kwargs.edgelist) != 0):
        print("Input graph path     :", kwargs.edgelist)
    elif(len(kwargs.adjlist) != 0):
        print("Input graph path     :", kwargs.adjlist)    
    print("Embedding dimensions :", kwargs.n_dim)
    print("Output directory     :", kwargs.outdir)
    print("Output filename      :", kwargs.filename)

    from .fd import embed_basic
    embeddings_df = embed_basic(**kwargs)

    save_embeddings(embeddings_df, filepath, kwargs.format)
    # End of fd_basic 
    #########

@common_options
@click.option('--k1', type=float, default=0.999, help='k1 parameter.')
@click.option('--k2', type=float, default=1.0,   help='k2 parameter.')
@click.option('--k3', type=float, default=10.0,  help='k3 parameter.')
@click.option('--k4', type=float, default=0.01,  help='k4 parameter.')
@click.option('--coeffs', nargs=4, type=float, help='Coefficients for the force calculation. If provided, overrides k1, k2, k3, k4. Used to shorten the syntanx.')
def fd_shell(**kwargs):
    """Generate embeddings using the force-directed basic algorithm with shell averaging."""
    kwargs = rns(kwargs)
    if(kwargs.coeffs is not None):
        kwargs.k1 = float(kwargs.coeffs[0])
        kwargs.k2 = float(kwargs.coeffs[1])
        kwargs.k3 = float(kwargs.coeffs[2])
        kwargs.k4 = float(kwargs.coeffs[3])
    
    print("fd-basic command with params:")
    for k,v in kwargs.items():
        print(f"{str(k):<16s}: {v}")
    
    # Make the filename
    filepath = os.path.join(kwargs.outdir, kwargs.filename)

    if(len(kwargs.edgelist) != 0):
        print("Input graph path     :", kwargs.edgelist)
    elif(len(kwargs.adjlist) != 0):
        print("Input graph path     :", kwargs.adjlist)    
    print("Embedding dimensions :", kwargs.n_dim)
    print("Output directory     :", kwargs.outdir)
    print("Output filename      :", kwargs.filename)

    from .fd import embed_shell
    embeddings_df = embed_shell(**kwargs)

    save_embeddings(embeddings_df, filepath, kwargs.format)
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
def fd_targets(**kwargs):
    """Generate embeddings using the force-directed algorithm with selective target nodes."""
    kwargs = rns(kwargs)
    if(kwargs.coeffs is not None):
        kwargs.k1 = kwargs.coeffs[0]
        kwargs.k2 = kwargs.coeffs[1]
        kwargs.k3 = kwargs.coeffs[2]
        kwargs.k4 = kwargs.coeffs[3]
        kwargs.reach_a = kwargs.coeffs[4]
        kwargs.reach_r = kwargs.coeffs[5]
        kwargs.landmarks_ratio = kwargs.coeffs[6]

    print("fd-landmark command with params:")
    for k,v in kwargs.items():
        print(f"{str(k):<16s}: {v}")
    
    filepath = os.path.join(kwargs.outdir, kwargs.filename)

    if(len(kwargs.edgelist) != 0):
        print("Input graph path     :", kwargs.edgelist)
    elif(len(kwargs.adjlist) != 0):
        print("Input graph path     :", kwargs.adjlist)    
    print("Embedding dimensions :", kwargs.n_dim)
    print("Output directory     :", kwargs.outdir)
    print("Output filename      :", kwargs.filename)

    from .fd import embed_targets
    embeddings_df = embed_targets(**kwargs)

    save_embeddings(embeddings_df, filepath, kwargs.format)
    # End of fd_targets
    #########
