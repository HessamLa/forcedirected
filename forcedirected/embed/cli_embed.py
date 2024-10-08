import click
@click.group()
def cli_embed():
    """Main entry point for the graph embedding CLI tool."""
    pass

import os
import functools
from recursivenamespace import rns
from .embed_utils import save_embeddings

import yaml
def load_config(ctx, config_path):
    options = {}
    if(config_path is None):
        return options
    with open(config_path, 'r') as f:
        config_params = yaml.safe_load(f)
        for k,v in config_params.items():
            parameter_source = ctx.get_parameter_source(k)
            # print(f"{k:<20s}: {parameter_source}")
            if(parameter_source is None):
                options[k] = v
            elif(parameter_source.name == "DEFAULT"):
                options[k] = v # override the default params
    return options

def check_required(options, required):
    missing_list=[]
    for r in required:
        if(options[r] is None):
            # print(f"'{r}' is a required options and must be provide, either through the command line or config file.")
            missing_list.append(r)
    return missing_list            
    
def common_options(func):
    """Decorator to apply common options to graph generation commands."""
    required_options = ['name', 'edgelist'] # maintain the required options in this list.
    
    @cli_embed.command(context_settings=dict(show_default=True))
    @click.option('--config', type=click.Path(exists=True), help='Path to the config file.')
    @click.option('-d', '--n-dim', '--ndim', type=int, default=128, help='Number of dimensions.')
    @click.option('--epochs', type=int, default=1000, help='Number of epochs.', show_default=True)
    @click.option('--lr', type=float, default=1.0, help='Learning rate.', show_default=True)
    @click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda']), default='auto', help='Device to use for computation.', show_default=True)
    @click.option('-n', '--name', type=str, help='Name of the graph. Required, either from the command line or the config file.') # required. input either from commmand line or the config file
    @click.option('-e', '--edgelist', type=click.Path(), help='Path to the edge list file. Either this or adjlist must be provided. Required, either from the command line or the config file.') # required. input either from commmand line or the config file
    @click.option('--outdir', type=click.Path(), default='./data', help='Output directory for the embeddings.', show_default=True)
    @click.option('--format',  type=click.Choice(['csv', 'pkl', 'pqt']), default='csv', help='Output file type. csv, Pandas pickle, or parquet.', show_default=True)
    @click.option('-z', '--filename', type=click.Path(), help='Output filename for the embeddings. [default: <dataset>-<method>-d<dim>.<format>]', show_default=False)

    @click.option('--verbosity', type=click.INT, default=2, show_default=True,
                help='Verbosity level as defined in ForceDirected base model. '
                    '0: no output, 1: essential msg, 2: short msg, 3: full msg + exception msg, '
                    '4: full msg + exception msg + raise exception.')
    @click.option('--seed', type=int, default=None, help='Random seed for reproducibility.', show_default=True)
    @click.pass_context
    @functools.wraps(func)
    def wrapper(ctx, *args, **options):
        options = rns(options)         
        # load the config file
        options.update(load_config(ctx, options.config))
        
        # check the required options
        missing_options = check_required(options, required_options)
        if(len(missing_options)>0):
            for m in missing_options:
                print(f"'{m}' is a required options and must be provide, either through the command line or config file.")
            exit(1)

        if(not os.path.exists(options.edgelist)):
            print(f"Edge list file not found: {options.edgelist}")
            exit(1)
        return func(*args, **options)
    return wrapper

def fd_base(**options):
    """Base function for forcedirected embedding."""
    options = rns(options)

    # change the 'model' argument to 'model_module'
    if ('model' in options):
        options.model_module = options.model
        del options.model

    # Make the filename
    if(options.filename is None):
        options.filename = f"{options.name}-{options.model_module}-d{options.n_dim}.{options.format}"
    
    print("command with params:")
    for k,v in options.items():
        print(f"{str(k):<16s}: {v}")
    
    print("Input graph path     :", options.edgelist)
    print("Embedding dimensions :", options.n_dim)
    print("Output directory     :", options.outdir)
    print("Output filename      :", options.filename)

    from .fd import embed
    embeddings_df = embed(**options)

    filepath = os.path.join(options.outdir, options.filename)
    save_embeddings(embeddings_df, filepath, options.format, set_column_names=True)

# @click.option('--model', 'model_module', type=str, help='Path to the embeddings module. Ex. model_201_basic.', show_default=True, required=True)
@click.option('--model', 'model_module', type=str, help='Path to the embeddings module. Ex. model_201_basic.', show_default=True)
@click.option('--k1', type=float, default=0.999, help='k1 parameter.')
@click.option('--k2', type=float, default=1.0,   help='k2 parameter.')
@click.option('--k3', type=float, default=1.0,  help='k3 parameter.')
@click.option('--k4', type=float, default=0.01,  help='k4 parameter.')
@click.option('--coeffs', nargs=4, type=float, help='Coefficients for the force calculation. If provided, overrides k1, k2, k3, k4. Used to shorten the syntanx.')
@common_options
def fd(**options):
    """FD algorithm, with the provided model."""
    options = rns(options)
    
    fd_base(**options)
    return
    # End of fd_basic 
    #########

@common_options
# @click.option('--model', type=str, default='model_201_basic', help='Path to the embeddings file.', show_default=True)
@click.option('--k1', type=float, default=0.999, help='k1 parameter.')
@click.option('--k2', type=float, default=1.0,   help='k2 parameter.')
@click.option('--k3', type=float, default=10.0,  help='k3 parameter.')
@click.option('--k4', type=float, default=0.01,  help='k4 parameter.')
@click.option('--coeffs', nargs=4, type=float, help='Coefficients for the force calculation. If provided, overrides k1, k2, k3, k4. Used to shorten the syntanx.')
def fdbasic(**options):
    """FD basic algorithm."""
    options = rns(options)
    # set the parameters
    if(options.coeffs is not None):
        options.k1 = float(options.coeffs[0])
        options.k2 = float(options.coeffs[1])
        options.k3 = float(options.coeffs[2])
        options.k4 = float(options.coeffs[3])
    # Make the filename
    if(options.filename is None):
        options.filename = f"{options.name}-fdbasic-d{options.n_dim}.{options.format}"

    from forcedirected.models import model_201_basic
    options.model_module = model_201_basic
    fd_base(**options)
    return
    # End of fd_basic 
    #########

@common_options
@click.option('--k1', type=float, default=0.999, help='k1 parameter.')
@click.option('--k2', type=float, default=1.0,   help='k2 parameter.')
@click.option('--k3', type=float, default=10.0,  help='k3 parameter.')
@click.option('--k4', type=float, default=0.01,  help='k4 parameter.')
@click.option('--coeffs', nargs=4, type=float, help='Coefficients for the force calculation. If provided, overrides k1, k2, k3, k4. Used to shorten the syntanx.')
def fdshell(**options):
    """FD basic algorithm with shell averaging."""
    options = rns(options)
    if(options.coeffs is not None):
        options.k1 = float(options.coeffs[0])
        options.k2 = float(options.coeffs[1])
        options.k3 = float(options.coeffs[2])
        options.k4 = float(options.coeffs[3])
    if(options.filename is None):
        options.filename = f"{options.name}-fdshell-d{options.n_dim}.{options.format}"

    from forcedirected.models import model_204_shell
    options.model_module = model_204_shell
    fd_base(**options)
    return
    # End of fdshell 
    #########

@common_options
@click.option('-k1','--k1',      type=float, default=0.999, help='k1 parameter.')
@click.option('-k2','--k2',      type=float, default=1.0,   help='k2 parameter.')
@click.option('-k3','--k3',      type=float, default=10.0,  help='k3 parameter.')
@click.option('-k4','--k4',      type=float, default=0.01,  help='k4 parameter.')
@click.option('-a', '--reach_a', type=int,   default=1,    help='Attractive reach parameter.')
@click.option('-r', '--reach_r', type=int,   default=4,    help='Repulsive reach parameter.')
@click.option('-L', '--landmarks_ratio', type=click.FloatRange(0, 1.0), default=0.01, help='Ratio of landmark nodes (top ratio of degrees).')
@click.option('--coeffs', nargs=7, # type=(float, float, float, float, int, int, click.FloatRange(0, 1.0)), 
              help='Coefficients for the force calculation. If provided, overrides other parameters. Used to shorten the syntanx.')
def fdtargets(**options):
    """FD algorithm with selective target nodes."""
    options = rns(options)
    # set the parameters
    if(options.coeffs is not None):
        options.k1 = float(options.coeffs[0])
        options.k2 = float(options.coeffs[1])
        options.k3 = float(options.coeffs[2])
        options.k4 = float(options.coeffs[3])
        options.reach_a = int(options.coeffs[4])
        options.reach_r = int(options.coeffs[5])
        options.landmarks_ratio = float(options.coeffs[6])
    if(options.filename is None):
        options.filename = f"{options.name}-fdtargets-d{options.n_dim}.{options.format}"

    from forcedirected.models import model_214_targets
    options.model_module = model_214_targets
    fd_base(**options)
    return
    # End of fdtargets
    #########

@common_options
@click.option('-k1','--k1',      type=float, default=0.999, help='k1 parameter.')
@click.option('-k2','--k2',      type=float, default=1.0,   help='k2 parameter.')
@click.option('-k3','--k3',      type=float, default=10.0,  help='k3 parameter.')
@click.option('-k4','--k4',      type=float, default=0.01,  help='k4 parameter.')
@click.option('-a', '--reach_a', type=int,   default=1,    help='Attractive reach parameter.')
@click.option('-r', '--reach_r', type=int,   default=4,    help='Repulsive reach parameter.')
@click.option('-L', '--landmarks_ratio', type=click.FloatRange(0, 1.0), default=0.01, help='Ratio of landmark nodes (top ratio of degrees).')
@click.option('--coeffs', nargs=7, #type=(float, float, float, float, int, int, click.FloatRange(0, 1.0)), 
              help='Coefficients for the force calculation. If provided, overrides other parameters. Used to shorten the syntanx.')
def fdtargets_mem(**options):
    """FD algorithm with selective target nodes (optimized memory utilization implementation)."""
    options = rns(options)
    # set the parameters
    if(options.coeffs is not None):
        options.k1 = float(options.coeffs[0])
        options.k2 = float(options.coeffs[1])
        options.k3 = float(options.coeffs[2])
        options.k4 = float(options.coeffs[3])
        options.reach_a = int(options.coeffs[4])
        options.reach_r = int(options.coeffs[5])
        options.landmarks_ratio = float(options.coeffs[6])
    if(options.filename is None):
        options.filename = f"{options.name}-fdtargets_mem-d{options.n_dim}.{options.format}"

    from forcedirected.models import model_215_targets_mem
    options.model_module = model_215_targets_mem
    fd_base(**options)
    return
    # End of fdtargets
    #########


@common_options
@click.option('-k1','--k1',      type=float, default=0.999, help='k1 parameter.')
@click.option('-k2','--k2',      type=float, default=1.0,   help='k2 parameter.')
@click.option('-k3','--k3',      type=float, default=10.0,  help='k3 parameter.')
@click.option('-k4','--k4',      type=float, default=0.01,  help='k4 parameter.')
@click.option('-a', '--reach_a', type=int,   default=1,    help='Attractive reach parameter.')
@click.option('-f', '--full_max', type=int,   default=4,    help='Full neighborhood maximum.')
@click.option('-t', '--tlog_coeff', type=float, default=100, help='Coefficient for log(n), with n being the number of nodes.')
@click.option('-L', '--landmarks_ratio', type=click.FloatRange(0, 1.0), default=0.01, help='Ratio of landmark nodes (top ratio of degrees).')
@click.option('--coeffs', nargs=8, # type=(float, float, float, float, int, int, click.FloatRange(0, 1.0)), 
              help='Coefficients for the force calculation. If provided, overrides other parameters. Used to shorten the syntanx.')
def fdlandmarks(**options):
    # """NON OPRATIONAL"""
    """FD algorithm with selective target nodes (optimized memory utilization implementation)."""
    # raise NotImplementedError("This function is not operational.")
    options = rns(options)
    # set the parameters
    if(options.coeffs is not None):
        options.k1 = float(options.coeffs[0])
        options.k2 = float(options.coeffs[1])
        options.k3 = float(options.coeffs[2])
        options.k4 = float(options.coeffs[3])
        options.reach_a = int(options.coeffs[4])
        options.full_max = int(options.coeffs[5])
        options.tlog_coeff = float(options.coeffs[6])
        options.landmarks_ratio = float(options.coeffs[7])
    if(options.filename is None):
        options.filename = f"{options.name}-fdlandmarks-d{options.n_dim}.{options.format}"

    from forcedirected.models import model_224_random_landmarks
    options.model_module = model_224_random_landmarks
    fd_base(**options)
    return
    # End of fdtargets
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
def node2vec(edgelist, **options):
    """Embed using the node2vec algorithm."""
    options = rns(options)

    if(options.filename is None):
        options.filename = f"{options.name}-node2vec-d{options.n_dim}.{options.format}"
    filepath = os.path.join(options.outdir, options.filename)

    print("node2vec command with params:")
    for k,v in options.items():
        print(f"{str(k):<16s}: {v}")

    print("Input graph path     :", options.edgelist)
    print("Embedding dimensions :", options.n_dim)
    print("Output directory     :", options.outdir)
    print("Output filename      :", options.filename)

    from .node2vec import embed_node2vec
    embeddings_df = embed_node2vec(edgelist, **options)

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
    """Embed using the GraphSAGE algorithm."""
    options = rns(options)

    if(options.filename is None):
        options.filename = f"{options.name}-graphsage-d{options.n_dim}.{options.format}"
    filepath = os.path.join(options.outdir, options.filename)

    print("graphsage command with params:")
    for k,v in options.items():
        print(f"{str(k):<16s}: {v}")

    if(len(options.edgelist) != 0):
        print("Input graph path     :", options.edgelist)
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
    """Embed using the LINE algorithm."""
    options = rns(options)

    if(options.filename is None):
        options.filename = f"{options.name}-line-d{options.n_dim}.{options.format}"
    filepath = os.path.join(options.outdir, options.filename)

    print("line command with params:")
    for k,v in options.items():
        print(f"{str(k):<16s}: {v}")

    print("Input graph path     :", options.edgelist)
    print("Embedding dimensions :", options.n_dim)
    print("Output directory     :", options.outdir)
    print("Output filename      :", options.filename)

    from .line import embed_line
    embeddings_df = embed_line(**options)

    save_embeddings(embeddings_df, filepath, options.format, set_column_names=True)
    # End of node2vec
    #########

# this is the new idea
@common_options
@click.option('--k1', type=float, default=0.999, help='k1 parameter.')
@click.option('--k2', type=float, default=1.0,   help='k2 parameter.')
@click.option('--k3', type=float, default=10.0,  help='k3 parameter.')
@click.option('--k4', type=float, default=0.01,  help='k4 parameter.')
@click.option('--coeffs', nargs=4, type=float, help='Coefficients for the force calculation. If provided, overrides k1, k2, k3, k4. Used to shorten the syntanx.')
def fdcandid(**options):
    """FD basic algorithm with candid nodes and shell averaging."""
    options = rns(options)
    if(options.coeffs is not None):
        options.k1 = float(options.coeffs[0])
        options.k2 = float(options.coeffs[1])
        options.k3 = float(options.coeffs[2])
        options.k4 = float(options.coeffs[3])
    if(options.filename is None):
        options.filename = f"{options.name}-fdcandid-d{options.n_dim}.{options.format}"

    from forcedirected.models import model_224_candid_targets
    options.model_module = model_224_candid_targets
    fd_base(**options)
    return
    # End of fdcandid 
    #########

