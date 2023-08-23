
# %%
# reload modules
# %load_ext autoreload
# %autoreload 2
# get arguments
import os, sys
import argparse
from typing import Any

from pprint import pprint

import numpy as np

import networkx, networkit # this line is for typing hints used in this code
import networkx as nx
import networkit as nk

from .callbacks import StatsLog
import torch

# %%
# %%


def process_arguments(
                # You can override the following default parameters by argument passing
                EMBEDDING_METHOD='forcedirected',
                DEFAULT_DATASET='ego-facebook',
                OUTPUTDIR_ROOT='./embeddings-tmp',
                DATASET_CHOICES=['tinygraph', 'cora', 'citeseer', 'pubmed', 'ego-facebook', 'corafull', 'wiki', 'blogcatalog', 'flickr', 'youtube'],
                # MODEL_VERSION_CHOICES=['1','2','3','104','4nodrop', '5', '5z2', '6', '7'],
                MODEL_VERSION_CHOICES=['104'],
                NDIM=128, ALPHA=0.3,
                ):
    
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-d', '--dataset', type=str, default=DEFAULT_DATASET, choices=DATASET_CHOICES, 
                        help='name of the dataset (default: cora)')
    parser.add_argument('-v', '--fdversion', type=str, default='104', choices=MODEL_VERSION_CHOICES,
                        help='version of the force-directed model (default: 104)')
    parser.add_argument('--outputdir_root', type=str, default=OUTPUTDIR_ROOT, 
                        help=f"Root output directory (default: {OUTPUTDIR_ROOT}")
    parser.add_argument('--outputdir', type=str, default=None, 
                        help=f"output directory (default: {{outputdir_root}}/{EMBEDDING_METHOD}_v{{version}}_{{ndim}}d/{DEFAULT_DATASET})")
    parser.add_argument('--outputfilename', type=str, default='embed-df.pkl', 
                        help='filename to save the final result (default: embed-df.pkl). Use pandas to open.')
    parser.add_argument('--historyfilename', type=str, default='embed-hist.pkl', 
                        help='filename to store s sequence of results from each iteration (default: embed-hist.pkl). Use pickle loader to open.')
    parser.add_argument('--save-history-every', type=int, default=100, 
                        help='save history every n iteration.')
    parser.add_argument('--save-stats-every', type=int, default=10, 
                        help='save history every n iteration.')
    parser.add_argument('--statsfilename', type=str, default='stats.csv', 
                        help='filename to save the embedding stats history (default: stats.csv)')
    parser.add_argument('--logfilepath', type=str, default=None, 
                        help='path to the log file (default: {EMBEDDING_METHOD}-{dataset}.log)')
    parser.add_argument('--ndim', type=int, default=NDIM, 
                        help=f'number of embedding dimensions (default: {NDIM})')
    parser.add_argument('--alpha', type=float, default=ALPHA,
                        help='alpha parameter (default: 0.3)')
    parser.add_argument('--epochs', type=int, default=5000, 
                        help='number of iterations for embedding process (default: 5000)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='choose the device to run the process on (default: auto). auto will use cuda if available, otherwise cpu.')
    parser.add_argument('--std0', type=float, default=1.0, 
                        help='initialization standard deviation (default: 1.0).')
    parser.add_argument('--base_std0', type=float, default=0.005, 
                        help='base standard deviation of noise (default: 0.005), noise_std = f(t)*std0 + base_std0. f(t) converges to 0.')
    parser.add_argument('--add-noise', action='store_true', 
                        help='enable noise')
    parser.add_argument('--random-drop', default='steady-rate', type=str,
                        choices=['steady-rate', 'exponential-diminish', 'linear-decrease', 'linear-increase', 'none', ''], 
                        help='Random drop mode')
    parser.add_argument('--random-drop-params', default=[0.5], type=float, nargs='+',
                        help='Random drop parameter values')

    parser.add_argument('--description', type=str, default="", nargs='+', 
                        help='description, used for experimentation logging')
    args, unknown = parser.parse_known_args()

    if(len(unknown)>0):
        print("====================================")
        print("THERE ARE UNKNOWN ARGUMENTS PASSED:")
        pprint(unknown)
        print("====================================")
        input("Enter a key to continue...")
        
    # # use default parameter values if required
    # if  (args.fdversion=='1'): args.FDModel = FDModel_1
    # elif(args.fdversion=='2'): args.FDModel = FDModel_2
    # elif(args.fdversion=='3'): args.FDModel = FDModel_3
    # elif(args.fdversion=='4'): args.FDModel = FDModel_4
    # elif(args.fdversion=='4nodrop'): args.FDModel = FDModel_4nodrop
    # elif(args.fdversion=='5'): args.FDModel = FDModel_5

        # use default parameter values if required
    if(args.fdversion=='104'):          from .models.model_104 import FDModel
    elif(args.fdversion=='4nodrop'):    from .models.model_4nodrop import FDModel
    elif(args.fdversion=='5'):          from .models.model_5 import FDModel
    elif(args.fdversion=='5z2'):          from .models.model_5z2 import FDModel
    elif(args.fdversion=='6'):          from .models.model_6 import FDModel
    elif(args.fdversion=='7'):          from .models.model_7 import FDModel
    args.FDModel = FDModel

    if(args.outputdir is None):
        args.outputdir = f"{args.outputdir_root}/{EMBEDDING_METHOD}_v{args.FDModel.VERSION}_{args.ndim}d/{args.dataset}"
    if(args.logfilepath is None):
        args.logfilepath = f"{args.outputdir}/{EMBEDDING_METHOD}-{args.dataset}.log"

    # Combine the description words into a single string
    args.description = ' '.join(args.description)

    args.emb_filepath = f"{args.outputdir}/{args.outputfilename}" # the path to store the latest embedding
    args.hist_filepath = f"{args.outputdir}/{args.historyfilename}" # the path to APPEND the latest embedding
    args.stats_filepath = f"{args.outputdir}/{args.statsfilename}" # the path to save the latest stats


    print("\nArguments:")
    for key, value in vars(args).items():
        print(f"{key:20}: {value}")
    print("")
    return args


if __name__ == '__main__':
    print("Current directory:", os.getcwd())
    # args = process_arguments(DEFAULT_DATASET='cora')
    # args = process_arguments(DEFAULT_DATASET='ego-facebook')
    # args = process_arguments(DEFAULT_DATASET='tinygraph')
    args = process_arguments()
    os.makedirs(args.outputdir, exist_ok=True)

    
    DATA_ROOT=os.path.expanduser('~/gnn/datasets')       # root of datasets to obtain graphs from
    # EMBEDS_ROOT=os.path.expanduser('~/gnn/embeddings')    # root of embeddings to save to
    if(args.dataset in ['cora', 'pubmed', 'citeseer', 'tinygraph', 'ego-facebook', 'corafull', 'blogcatalog']):
        args.nodelist = f'{DATA_ROOT}/{args.dataset}/{args.dataset}_nodes.txt'
        args.features = f'{DATA_ROOT}/{args.dataset}/{args.dataset}_x.txt'
        args.labels   = f'{DATA_ROOT}/{args.dataset}/{args.dataset}_y.txt'
        args.edgelist = f'{DATA_ROOT}/{args.dataset}/{args.dataset}_edgelist.txt'
    elif(args.dataset in ['wiki']):
        args.nodelist = f'{DATA_ROOT}/wiki/Wiki_nodes.txt'
        args.features = f'{DATA_ROOT}/wiki/Wiki_category.txt'
        args.labels   = f'{DATA_ROOT}/wiki/Wiki_labels.txt'
        args.edgelist = f'{DATA_ROOT}/wiki/Wiki_edgelist.txt'
    elif(args.dataset in ['blogcatalog']):
        args.nodelist = f'{DATA_ROOT}/BlogCatalog-dataset/data/nodes.csv'
        args.edgelist = f'{DATA_ROOT}/BlogCatalog-dataset/data/edges.csv'
        args.labels   = f'{DATA_ROOT}/BlogCatalog-dataset/data/labels.csv' # the first column is the node id, the second column is the label

    # load graph from files into a networkx object
    def load_graph(args: dict()):
        ##### LOAD GRAPH #####
        Gx = nx.Graph()
        # load nodes first to keep the order of nodes as found in nodes or labels file
        if('nodelist' in args and os.path.exists(args.nodelist)):
            Gx.add_nodes_from(np.loadtxt(args.nodelist, dtype=str, usecols=0))
            print('loaded nodes from nodes file')
        elif('features' in args and  os.path.exists(args.features)):
            Gx.add_nodes_from(np.loadtxt(args.features, dtype=str, usecols=0))
            print('loaded nodes from features file')
        elif('labels' in args and os.path.exists(args.labels)):
            Gx.add_nodes_from(np.loadtxt(args.labels, dtype=str, usecols=0))   
            print('loaded nodes from labels file')
        else:
            print('no nodes were loaded from files. nodes will be loaded from edgelist file.')

        # add edges from args.path_edgelist
        Gx.add_edges_from(np.loadtxt(args.edgelist, dtype=str))
        return Gx
    Gx = load_graph(args)

    print("Number of nodes:", Gx.number_of_nodes())
    print("Number of edges:", Gx.number_of_edges())
    print("Number of connected components:", nx.number_connected_components(Gx))    

    # Y = read_labels(args.path_labels)

    if(args.device == 'auto'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    from forcedirected.utilityclasses import Callback_Base
    class EarlyStopping (Callback_Base):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)

        def on_epoch_end(self, fd_model, epoch, **kwargs):
            Fa = torch.norm(fd_model.fmodel_attr.F, dim=1)
            Fa = torch.sum(Fa).item()
            Fr = torch.norm(fd_model.fmodel_repl.F, dim=1)
            Fr = torch.sum(Fr).item()
            # if absolute difference of Fa and Fr is less than 1e-3, stop training
            if(abs(Fa-Fr)<1e-3):
                fd_model.stop_training = True
                print("Early Stopping")
                
    ##### START EMBEDDING #####    
    model = args.FDModel(Gx, n_dims=args.ndim, alpha=args.alpha, callbacks=[StatsLog(args=args), EarlyStopping()])
    model.train(epochs=args.epochs, device=device)
    with open(args.outputdir+'/done.txt', 'w') as f:
        f.write('done')
    print(f'Embedding completed, dataset={args.dataset}, version={args.fdversion}, ndim={args.ndim}')
    print(f'Embedding completed, dataset={args.dataset}, version={args.fdversion}, ndim={args.ndim}', file=sys.stderr)
    ##### SAVE EMBEDDINGS #####
    embeddings = model.get_embeddings()
    

    # pos_dict = dict(zip(Gx.nodes(), model.Z[:,:2].cpu().numpy()))
    # nx.draw(Gx, pos=pos_dict, node_size=10, width=0.1, node_color='black', edge_color='gray')


# %%
