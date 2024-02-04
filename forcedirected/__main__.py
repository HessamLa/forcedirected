
# %%
# reload modules
# %load_ext autoreload
# %autoreload 2
# get arguments
import os, sys
import argparse
from typing import Any

import importlib

from pprint import pprint

import numpy as np

import networkx, networkit # this line is for typing hints used in this code
import networkx as nx
import networkit as nk

from .callbacks import StatsLog, SaveEmbedding
import torch

# %%
def try_import_module(module_name, attr_name=None):
    try:
        # Import the module dynamically
        model_module = importlib.import_module(module_name, package=__package__)
        if attr_name is None:
            return model_module
        else:
            return getattr(model_module, attr_name)
    except (ModuleNotFoundError, AttributeError) as e:
        # Return None if the module or FDModel class is not found
        return None

def process_arguments(
                # You can override the following default parameters by argument passing
                EMBEDDING_METHOD='forcedirected',
                DEFAULT_DATASET='ego-facebook',
                OUTPUTDIR_ROOT='./embeddings-tmp',
                DATASET_CHOICES=['tinygraph', 'cora', 'citeseer', 'pubmed', 'ego-facebook', 'corafull', 'wiki', 'blogcatalog', 'flickr', 'youtube'],
                # MODEL_VERSION_CHOICES=[ '104', '104nodrop',
                #                         '106', '107', '108', '109', '110', '110k100',
                #                         '111nodrop',
                #                         '120', '121', 
                #                         '201',
                #                         '301',
                #                         ],
                NDIM=128, ALPHA=0.3,
                ):
    
    # edgelist=args.edgelist, nodelist=args.nodelist, features=args.features, labels=args.labels
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-d', '--dataset_name', type=str, required=True, choices=DATASET_CHOICES, 
                        help='Name of the dataset. This is used to create the output paths.')
    parser.add_argument('-e', '--edgelist', type=str, required=True,
                        help='path to the edgelist file')
    parser.add_argument('--nodelist', type=str, default='',
                        help='path to the nodelist file')
    parser.add_argument('--features', type=str, default='',
                        help='path to the features file')
    parser.add_argument('--labels', type=str, default='',
                        help='path to the labels file')
    parser.add_argument('-v', '--fdversion', type=str, default='104',
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
    parser.add_argument('--lr', type=float, default=1.0, 
                        help='learning rate for updating the gradient (default: 1.0)')
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

    # if(args.fdversion=='104'):          from .models.model_104 import FDModel
    # elif(args.fdversion=='104nodrop'):
    #                                     from .models.model_104nodrop import FDModel
    # elif(args.fdversion=='106'):          from .models.model_106 import FDModel
    # elif(args.fdversion=='107'):          from .models.model_107 import FDModel
    # elif(args.fdversion=='108'):          from .models.model_108 import FDModel
    # elif(args.fdversion=='109'):          from .models.model_109 import FDModel
    # elif(args.fdversion=='110'):          from .models.model_110 import FDModel
    # elif(args.fdversion=='110k100'):      from .models.model_110k100 import FDModel
    # elif(args.fdversion=='111nodrop'):      from .models.model_111nodrop import FDModel
    # elif(args.fdversion=='120'):          from .models.model_120 import FDModel
    # elif(args.fdversion=='121'):          from .models.model_121 import FDModel
    # elif(args.fdversion=='201'):          from .models.model_201 import FDModel
    # elif(args.fdversion=='301'):          from .models.model_301 import FDModel
    # load module from .models.model_xxx import FDModel such that xxx is from args.fdversion
    # args.FDModel = FDModel

    # Try importing from the first pattern
    print(f"Trying to import 'FDModel' from .models.model_{args.fdversion}")
    args.FDModel = try_import_module(f'.models.model_{args.fdversion}', 'FDModel')
    if(args.FDModel is None):
        raise ValueError(f"Failed to import 'FDModel' version {args.fdversion} from .models.model_{args.fdversion}")

    if(args.outputdir is None):
        args.outputdir = f"{args.outputdir_root}/{EMBEDDING_METHOD}_v{args.FDModel.VERSION}_{args.ndim}d/{args.dataset_name}"
    if(args.logfilepath is None):
        args.logfilepath = f"{args.outputdir}/{EMBEDDING_METHOD}-{args.dataset_name}.log"

    # Combine the description words into a single string
    args.description = ' '.join(args.description)

    args.emb_filepath = f"{args.outputdir}/{args.outputfilename}" # the path to store the latest embedding
    args.hist_filepath = f"{args.outputdir}/{args.historyfilename}" # the path to APPEND the latest embedding
    args.stats_filepath = f"{args.outputdir}/{args.statsfilename}" # the path to save the latest stats


    print("\nArguments:")
    for key, value in vars(args).items():
        print(f"{key:20}: {value}")

    return args

# load graph from files into a networkx object
def load_graph(edgelist, nodelist=None, features=None, labels=None):
    print(os.path.exists(edgelist), edgelist)
    print(os.path.exists(nodelist), nodelist)
    print(os.path.exists(features), features)
    print(os.path.exists(labels), labels)
    ##### LOAD GRAPH #####
    Gx = nx.Graph()
    # load nodes first to keep the order of nodes as found in nodes or labels file
    if(os.path.exists(nodelist)):
        Gx.add_nodes_from(np.loadtxt(nodelist, dtype=str, usecols=0))
        print('loaded nodes from nodes file')
    elif(os.path.exists(features)):
        Gx.add_nodes_from(np.loadtxt(features, dtype=str, usecols=0))
        print('loaded nodes from features file')
    elif(os.path.exists(labels)):
        Gx.add_nodes_from(np.loadtxt(labels, dtype=str, usecols=0))   
        print('loaded nodes from labels file')
    else:
        print('no nodes were loaded from files. nodes will be loaded from edgelist file.')

    # add edges from paths.path_edgelist
    Gx.add_edges_from(np.loadtxt(edgelist, dtype=str))
    return Gx

def main():
    print("Current directory:", os.getcwd())
    # args = process_arguments(DEFAULT_DATASET='cora')
    # args = process_arguments(DEFAULT_DATASET='ego-facebook')
    # args = process_arguments(DEFAULT_DATASET='tinygraph')
    args = process_arguments()
    os.makedirs(args.outputdir, exist_ok=True)
    

    Gx = load_graph(edgelist=args.edgelist, nodelist=args.nodelist, features=args.features, labels=args.labels)
    if(Gx.number_of_nodes()==0):
        print("Graph is empty")
        exit()

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
            # Fmean=torch.zeros_like(fd_model.Z)
            # for F in model.forcev.values():
            #     Fmean += F.v
            # Fmean = torch.mean(torch.norm(Fmean, dim=1)).item()
            # print(f"Early stop check: Fmean={Fmean:.3f}")
            dZmean = torch.mean(torch.norm(fd_model.dZ, dim=-1)).item()
            # print(f"Early stop check: dZ_mean={dZmean:.3f}")
            # if absolute difference of Fa and Fr is less than 1e-3, stop training
            if(abs(dZmean)<1e-4):
                fd_model.stop_training = True
                print("Early Stopping")
                
    ##### START EMBEDDING #####
    callbacks = [StatsLog(args=args), EarlyStopping(), 
                 SaveEmbedding(args=args),
                 ]
    # callbacks = [StatsLog(args=args)]
    model = args.FDModel(Gx, n_dim=args.ndim, alpha=args.alpha, lr=args.lr, callbacks=callbacks)
    model.train(epochs=args.epochs, device=device)
    # for epoch in range(args.epochs):
    #     model.train(epochs=1, device=device)
    #     if(model.stop_training):
    #         break
    
    with open(args.outputdir+'/done.txt', 'w') as f:
        f.write('done')
    print(f'Embedding completed, dataset={args.dataset_name}, version={args.fdversion}, ndim={args.ndim}')
    print(f'Embedding completed, dataset={args.dataset_name}, version={args.fdversion}, ndim={args.ndim}', file=sys.stderr)
    ##### SAVE EMBEDDINGS #####
    embeddings = model.get_embeddings()
    

    # pos_dict = dict(zip(Gx.nodes(), model.Z[:,:2].cpu().numpy()))
    # nx.draw(Gx, pos=pos_dict, node_size=10, width=0.1, node_color='black', edge_color='gray')
if __name__=='__main__':
    main()

# %%
