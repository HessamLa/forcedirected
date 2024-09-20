import networkx as nx
import numpy as np
import pandas as pd
import os

from recursivenamespace import recursivenamespace

# load graph from files into a networkx object
def load_graph(edgelist, nodelist='', features='', labels='', **kwargs):
    ##### LOAD GRAPH #####
    Gx = nx.Graph()
    # load nodes first to keep the order of nodes as found in nodes or labels file
    if(os.path.exists(nodelist)):
        Gx.add_nodes_from(np.loadtxt(nodelist, dtype=str, usecols=0))
        print('Loaded nodes from nodes file')
    elif(os.path.exists(features)):
        Gx.add_nodes_from(np.loadtxt(features, dtype=str, usecols=0))
        print('Loaded nodes from features file')
    elif(os.path.exists(labels)):
        Gx.add_nodes_from(np.loadtxt(labels, dtype=str, usecols=0))   
        print('Loaded nodes from labels file')
    else:
        print('Nodes will ONLY be loaded from edgelist file.')

    # add edges from args.path_edgelist
    Gx.add_edges_from(np.loadtxt(edgelist, dtype=str))
    return Gx

def load_embeddings(path_embedding=''):
    if(path_embedding.endswith('.pkl')):
        import pickle
        with open(path_embedding, 'rb') as f:
            embed = pickle.load(f)
    elif(path_embedding.endswith('.npy')):
        embed = np.load(path_embedding)
    return embed

# load stats into a pandas dataframe object
def load_stats(path_stats=''):
    if(path_stats.endswith('.pkl')):
        import pickle
        with open(path_stats, 'rb') as f:
            stats = pickle.load(f)
    elif(path_stats.endswith('.csv')):
        import pandas as pd
        stats = pd.read_csv(path_stats)
    return stats

def load_labels(path_labels=''):
    if(path_labels.endswith('.pkl')):
        import pickle
        with open(path_labels, 'rb') as f:
            labels = pickle.load(f)
    elif(path_labels.endswith('.csv')):
        import pandas as pd
        labels = pd.read_csv(path_labels)
    else:
        with open(path_labels, 'r') as f:
            labels={}
            for line in f.readlines():
                line = line.replace(',', ' ')
                line = line.strip().split()
                labels[line[0]] = line[1]
    return labels

def read_csv(file_path, has_header=None):
    """
    Read a CSV file and return the data, delimiter, and header presence. Then convert the data to a Pandas dataframe and returns it.
    If has_header is None, the function will attempt to detect the header presence. Otherwise, it will use the provided boolean value.
    """
    import csv
    if(has_header is not None):
        if(not isinstance(has_header, bool)):
            raise ValueError("has_header must be None or a boolean value.")

    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        content = file.readline(4096)  # Read a portion of the file for sniffing
        file.seek(0)  # Reset file pointer to the beginning
        
        sniffer = csv.Sniffer()
        # Detect the CSV dialect (which includes the delimiter)
        try:
            dialect = sniffer.sniff(content)
        except csv.Error:
            # If the dialect cannot be determined, fall back to default settings
            print("Could not determine file dialect. Falling back to comma delimiter.")
            dialect = csv.excel()  # Default CSV dialect in Python assumes comma as delimiter
        
        if(has_header is None):
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
            # print("No headers detected.")
            pass

        # Read and print the rest of the rows
        data = list(reader)
        # convert data to a Pandas dataframe
        df = pd.DataFrame(data, columns=headers)
        return df
    # End of read_csv
