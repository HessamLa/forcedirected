import click
@click.group()
def cli_eval():
    """CLI tool for evaluation of the graph embeddings."""
    pass

import os
import functools
import pandas as pd
import numpy as np
import networkx as nx

from forcedirected.utilities import load_graph, read_csv
from recursivenamespace import rns

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
    required_options = ['path_embeddings'] # maintain the required options in this list.
    @cli_eval.command(context_settings=dict(show_default=True))
    @click.option('--config', type=click.Path(exists=True), default=None, help='Path to the config file.')
    @click.option('-n', '--name', type=str, help='Name of the graph.', required=False)
    @click.option('-z', '--path_embeddings', type=click.Path(), help='Path to the graph embedding.')
    @click.option('--embeddings-format', 'fmt_emb', type=click.Choice(['csv', 'pkl', 'pqt']), default='csv', help='Format of the embeddings file.', show_default=True)    
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

        # check if the embeddings file exists
        if(not os.path.exists(options.path_embeddings) or not os.path.isfile(options.path_embeddings)):
            print(f"Embeddings file not found: {options.path_embeddings}")
            exit(1)
        return func(*args, **options)
    return wrapper
    # End of common_options

def load_embeddings(filepath, format):
    try:
        if(format=='csv'):
            embeddings_df = pd.read_csv(filepath)
        elif(format=='pkl'):
            embeddings_df = pd.read_pickle(filepath)
        elif(format=='pqt'):
            embeddings_df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unknown embeddings format: {format}")
    except Exception as e:
        raise(f"Error loading embeddings from {filepath}. {e}")
    print(f"Embeddings loaded from {filepath}.")
    return embeddings_df
    # End of load_embeddings


@common_options
@click.option('-y', '--path_label', type=click.Path(), show_default=False, help='Path to the labels file. Labels are in a space-separated file with the first column as the node id (str) and last column as the label.')
@click.option('--test-ratio', '--test-size', 'test_size', type=click.FloatRange(0.0, 1.0), default=0.5, help='Ratio of the test size.')
@click.option('--train-ratio', '--train-size', 'train_size', type=click.FloatRange(0.0, 1.0), default=None, help='Ratio of the train size.')
@click.option('--classifier', 'classification_model', type=click.Choice(['rf', 'lr', 'knn', 'mlp', 'ada', 'svc', 'dt']), default='rf', show_default=True, 
                help='Classifier to use for node classification. rf: Random Forest, lr: Logistic Regression, knn: K-Nearest Neighbors, mlp: Multi-Layer Perceptron, ada: AdaBoost, svc: Support Vector Classifier, dt: Decision Tree.')
def nc(**options):
    """Evaluate the node classification performance of the embeddings."""
    options = rns(options)
    if(not os.path.exists(options.path_label)):
        print(f"Labels file not found: {options.path_label}")
        exit(1)
    if(options.train_size is None):
        options.train_size = 1 - options.test_size
    else:
        options.test_size = 1 - options.train_size

    print("Node Classification evaluation command with params:")
    for k,v in options.items():
        print(f"{str(k):<16s}: {v}")

    # load the embeddings from options.path_embeddings
    embeddings_df = load_embeddings(options.path_embeddings, options.fmt_emb)
    
    # include a 'id' column if not present
    if('id' not in embeddings_df.columns):
        raise ValueError("Embeddings file must have an 'id' column which corresponds each row to a node.")
    embeddings_df['id'] = embeddings_df['id'].astype(str)
    

    # load the labels from options.path_label according to the format
    y = read_csv(options.path_label, has_header=False)
    y.rename(columns={y.columns[0]: 'id'}, inplace=True)
    # rename the first column to 'id'
    y['id'] = y['id'].astype(str)
    
    # sort based on 'id' column
    y.sort_values('id', inplace=True)
    embeddings_df.sort_values('id', inplace=True)

    # remove the entries in y that are not in embeddings
    y = y[y['id'].isin(embeddings_df['id'])]
    y_ids = set(y['id'])
    e_ids = set(embeddings_df['id'])
    if(not y_ids == e_ids):
        print("ERROR: The nodes in the labels and the embeddings do not match.")
        # # show the difference
        # print(f"Labels: {y_ids.difference(e_ids)}")
        # print(f"Embeddings: {e_ids.difference(y_ids)}")
        exit(1)

    from .node_classification import eval_nc
    results = eval_nc(embeddings_df.iloc[:,1:], y.iloc[:,1:], **options)

    print("Node classification results:")
    for k,v in results.items():
        if(k=='confusion'):
            # print(f"{str(k):<16s}:\n{v}")
            pass # for now, don't show the confusion matrix
        else:
            if(isinstance(v, float)): v = f"{v:.4f}"
            elif(isinstance(v, int)): v = str(v)
            print(f"{str(k):<16s}: {v}")
    # End of eval_nc
    #########

@common_options
@click.option('-e', '--edgelist', type=click.Path(), help='Path to the edge list file. Either this or adjlist must be provided. (Somewhat required)')
# @click.option('--adjlist', type=click.Path(), help='Path to the adjacency list. Either this or edgelist must be provided.')
@click.option('--test-ratio', '--test-size','test_size', type=click.FloatRange(0.0, 1.0), default=0.5, help='Ratio of the test size.')
@click.option('--train-ratio', '--train-size', 'train_size', type=click.FloatRange(0.0, 1.0), default=None, help='Ratio of the test size.')
@click.option('--product', type=click.Choice(['concat', 'hadamard', 'average']), default='hadamard', help='Ratio of the test size.')
@click.option('--classifier', 'classifier', type=click.Choice(['rf', 'lr', 'knn', 'mlp', 'ada', 'svc', 'dt']), default='rf', show_default=True, 
                help='Classifier to use for node classification. rf: Random Forest, lr: Logistic Regression, knn: K-Nearest Neighbors, mlp: Multi-Layer Perceptron, ada: AdaBoost, svc: Support Vector Classifier, dt: Decision Tree.')
# @click.option('--negative-sampling-mode', 'negative_sampling_mode', type=click.Choice(['random', 'adamic', 'jaccard', 'preferential', 'community']), default='random', show_default=True, help='Negative sampling mode.')
@click.option('--negative-sampling-mode', 'negative_sampling_mode', type=str, default='random', show_default=True, help='Negative sampling mode.')
@click.option('--top-k', type=int, default=100, show_default=True, help='Top k nodes to consider for evaluation.')
# @click.option('--fit-transform', is_flag=True, help='Fit and transform the embeddings using StandardScaler.')
def lp(**options):
    """Evaluate the link prediction performance of the embeddings."""
    options = rns(options)
    if(not os.path.exists(options.edgelist)):
        print(f"Edgelist file not found: {options.path_embeddings}")
        exit(1)

    if(options.train_size is None):
        options.train_size = 1.0 - options.test_size
    else:
        options.test_size = 1.0 - options.train_size

    print("Link Prediction evaluation command with params:")
    for k,v in options.items():
        print(f"{str(k):<16s}: {v}")

    # load the embeddings from options.path_embeddings
    embeddings_df = load_embeddings(options.path_embeddings, options.fmt_emb)

    # include a 'id' column if not present
    if('id' not in embeddings_df.columns):
        raise ValueError("Embeddings file must have an 'id' column which corresponds each row to a node.")
    embeddings_df['id'] = embeddings_df['id'].astype(str)
    
    # load the graph
    Gx = load_graph(edgelist=options.edgelist)
    if(Gx.number_of_nodes()==0):
        print("Graph is empty")
        exit(1)
    print("graph loaded")
    print("Number of nodes:", Gx.number_of_nodes())
    print("Number of edges:", Gx.number_of_edges())
    print("Number of connected components:", nx.number_connected_components(Gx))    

    from .link_prediction import eval_lp
    # def eval_lp(graph, embeddings, train_size:float=0.5, test_size:float=None, classification_model:str='rf', metrics='all', negative_sampling_mode='random', top_k=100, seed=None, **kwargs):
    # results = eval_lp(Gx, embeddings.iloc[:,1:], test_size=options.test_size, classification_model=options.classifier)
    results = eval_lp(Gx, embeddings_df, classification_model=options.classifier, **options)

    # # load the labels from options.path_label according to the format
    # if options.path_label:
    #     y = read_csv(options.path_label)
    #     y.rename(columns={y.columns[0]: 'id'}, inplace