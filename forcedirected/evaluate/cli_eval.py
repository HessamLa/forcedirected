import os
import click
import functools
import pandas as pd
import numpy as np

from recursivenamespace import rns

@click.group()
def cli_eval():
    """Main entry point for the graph generation CLI tool."""
    pass

def common_options(func):
    """Decorator to apply common options to graph generation commands."""
    @cli_eval.command(context_settings=dict(show_default=True))
    @click.option('-n', '--name', type=str, help='Name of the graph.', required=False)
    @click.option('-z', '--path_embeddings', type=click.Path(), help='Path to the graph embedding.', required=True)
    @click.option('--embeddings-format', 'fmt_emb', type=click.Choice(['csv', 'pkl']), default='csv', help='Format of the embeddings file.', show_default=True)
    
    @click.option('-e', '--edgelist', type=click.Path(), help='Path to the edge list file. Either this or adjlist must be provided. (Somewhat required)')
    @click.option('--adjlist', type=click.Path(), help='Path to the adjacency list. Either this or edgelist must be provided.')
    
    @click.option('--verbosity', type=click.INT, default=2, show_default=True,
                help='Verbosity level as defined in ForceDirected base model. '
                    '0: no output, 1: essential msg, 2: short msg, 3: full msg + exception msg, '
                    '4: full msg + exception msg + raise exception.')
    @click.option('--seed', type=int, default=None, help='Random seed for reproducibility.', show_default=True)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
    # End of common_options

def read_csv(file_path):
    """Read a CSV file and return the data, delimiter, and header presence. Then convert the data to a Pandas dataframe and returns it."""
    import csv
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        content = file.read(4096)  # Read a portion of the file for sniffing
        file.seek(0)  # Reset file pointer to the beginning
        
        sniffer = csv.Sniffer()
        # Detect the CSV dialect (which includes the delimiter)
        try:
            dialect = sniffer.sniff(content)
        except csv.Error:
            # If the dialect cannot be determined, fall back to default settings
            print("Could not determine file dialect. Falling back to comma delimiter.")
            dialect = csv.excel()  # Default CSV dialect in Python assumes comma as delimiter
        
        # Check if the first row appears to be a header
        has_header = sniffer.has_header(content)

        # Read the file using detected dialect and header presence
        file.seek(0)  # Reset file pointer again if we're going to read further
        reader = csv.reader(file, dialect)
        
        headers = None
        if has_header:
            headers = next(reader)  # Extract headers
            print("Detected headers:", headers)
        else:
            print("No headers detected.")

        # Read and print the rest of the rows
        data = list(reader)
        # convert data to a Pandas dataframe
        df = pd.DataFrame(data, columns=headers)
        # return data, dialect.delimiter, has_header
        return df
    # End of read_csv

def load_embeddings(filepath, format):
    try:
        if(format=='csv'):
            embeddings = pd.read_csv(filepath)
        elif(format=='pkl'):
            embeddings = pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unknown embeddings format: {format}")
    except Exception as e:
        print(f"Error loading embeddings from {filepath}")
        print(e)
    print(f"Embeddings loaded from {filepath}")
    return embeddings
    # End of load_embeddings


@common_options
@click.option('-y', '--path_label', type=click.Path(), show_default=False, help='Path to the labels file. Labels are in a space-separated file with the first column as the node id (str) and last column as the label.')
@click.option('--classifier', 'classification_model', type=click.Choice(['rf', 'lr', 'knn', 'mlp', 'ada']), default='rf', show_default=True, 
                help='Classifier to use for node classification. rf: Random Forest, lr: Logistic Regression, knn: K-Nearest Neighbors, mlp: Multi-Layer Perceptron, ada: AdaBoost.')
def nc(**options):
    """Evaluate the node classification performance of the embeddings."""
    options = rns(options)
    print("Node classification evaluation command with params:")
    for k,v in options.items():
        print(f"{str(k):<16s}: {v}")

    # load the embeddings from options.path_embeddings
    embeddings = load_embeddings(options.path_embeddings, options.fmt_emb)
    # set the 'id' column to string
    embeddings['id'] = embeddings['id'].astype(str)
    

    # load the labels from options.path_label according to the format
    if options.path_label:
        y = read_csv(options.path_label)
        y.rename(columns={y.columns[0]: 'id'}, inplace=True)
        y['id'] = y['id'].astype(str)
    else:
        print("No labels provided. Skipping node classification evaluation.")
        return
    # rename the first column to 'id'
    

    # sort based on 'id' column
    y.sort_values('id', inplace=True)
    embeddings.sort_values('id', inplace=True)
    

    from .node_classification import eval_nc
    results = eval_nc(embeddings.iloc[:,1:], y.iloc[:,1:])

    print("Node classification results:")
    for k,v in results.items():
        if(k=='confusion'):
            print(f"{str(k):<16s}:\n{v}")
        else:
            print(f"{str(k):<16s}: {v}")
    # End of eval_nc
    #########
