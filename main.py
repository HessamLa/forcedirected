
# %%
# reload modules
%load_ext autoreload
%autoreload 2

# get arguments
import argparse
import os
import pickle
from typing import Any
# from types import *

from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx, networkit # this line is for typing hints used in this code
import networkx as nx
import networkit as nk


from forcedirected.utilities.graphtools import load_graph_networkx, process_graph_networkx
from forcedirected.utilities.reportlog import ReportLog

from forcedirected.algorithms import get_alpha_hops, pairwise_difference
from forcedirected.Functions import attractive_force_ahops, repulsive_force_hops_exp

from forcedirected.Functions import DropSteadyRate, DropLinearChange, DropExponentialDimish
from forcedirected.Functions import generate_random_points
from forcedirected.utilityclasses import ForceClass, NodeEmbeddingClass
from forcedirected.utilityclasses import Model_Base, Callback_Base
import torch

# %%

# from model_1 import ForceDirected
# from model_2 import FDModel
# from model_3 import FDModel
from model_4 import FDModel

def make_hops_stats(N, hops, maxhops):
    """N is pairwise euclidean distance, numpy/torch tensor
    hops is pairwise hop distance, numpy/torch tensor
    N and hops must have the same shape
    """
    if(isinstance(N, torch.Tensor)):
        get_mean = lambda x: x.mean().item()
        get_std  = lambda x: x.std().item()
        get_max  = lambda x: x.max().item()
    else:
        get_mean = lambda x: x.mean()
        get_std  = lambda x: x.std()
        get_max  = lambda x: x.max()
        
    s = {}
    maxhops = int(maxhops)
    # connected components
    for h in range(1, maxhops+1):
        tN = N[ hops==h ]
        if( len( tN )>0 ):
            s[f"hops{h}_mean"] = get_mean( tN )
            s[f"hops{h}_std"]  = get_std( tN )
        else:
            s[f"hops{h}_mean"] = None
            s[f"hops{h}_std"]  = None
    
    # for l in range(1, int(max(ht)+1)):
    #     mask = hops==l
    #     if(len(Nt[mask])>0):
    #         s[f"hops{l}_mean"] = mean(tN)
    #         s[f"hops{l}_std"]  = std(tN)
    #     # print(f"{l:3d} {Nt[mask].mean():10.3f} {Nt[mask].std():10.3f} {len(Nt[mask])/2:8.0f}")
    
    # disconnected components
    tN = N[hops>maxhops]
    if( len( tN )>0 ):
        s[f"hopsinf_mean"] = get_mean( tN )
        s[f"hopsinf_std"]  = get_std( tN )
    else:
        s[f"hopsinf_mean"] = None
        s[f"hopsinf_std"]  = None
        
    return s

def make_stats_log(model, epoch):
    logstr = ''
    s = {'epoch': epoch}

    s.update(make_hops_stats(model.N, model.hops, model.maxhops))

    summary_stats = lambda x: (torch.sum(x).item(), torch.mean(x).item(), torch.std(x).item())

    # attractive forces
    Fa = model.fmodel_attr.F
    s['fa-sum'],  s['fa-mean'],  s['fa-std']  = summary_stats( torch.norm(Fa, dim=1) )
    # weighted attractive forces
    s['wfa-sum'], s['wfa-mean'], s['wfa-std'] = summary_stats( torch.norm(Fa, dim=1)*model.degrees )

    # repulsive forces
    Fr = model.fmodel_repl.F
    s['fr-sum'],  s['fr-mean'],  s['fr-std']  = summary_stats( torch.norm(Fr, dim=1) )
    # weighted repulsive forces
    s['wfr-sum'], s['wfr-mean'], s['wfr-std'] = summary_stats( torch.norm(Fr, dim=1)*model.degrees )
    
    
    # sum of all forces, expected to converge to 0
    s['f-all'] = torch.norm( torch.sum(Fa+Fr, dim=0) ) 
    
    # sum of norm/magnitude of all forces, expected to converge
    s['f-all-sum'] = s['fa-sum'] + s['fr-sum']

    # sum of all weighted forces, expected to converge to 0
    s['wf-all'] = torch.norm( torch.sum( (Fa+Fr)*model.degrees[:, None], dim=0 ) )    

    # sum of norm/magnitude of all weighted forces, expected to converge
    s['wf-all-sum'] = s['wfa-sum'] + s['wfr-sum']

    # relocations
    s['relocs-sum'], s['relocs-mean'], s['relocs-std'] = summary_stats( torch.norm(model.dZ, dim=1) )
    
    # weighted relocations
    s['wrelocs-sum'], s['wrelocs-mean'], s['wrelocs-std'] = summary_stats( torch.norm(model.dZ, dim=1)*model.degrees )
    
    # convert all torch.Tensor elements to regular python numbers
    for k,v in s.items():
        if(type(v) is torch.Tensor):
            s[k] = v.item()

    logstr = ''
    logstr+= f"attr:{s['fa-sum']:<9.3f}({s['fa-mean']:.3f})  "
    logstr+= f"repl:{s['fr-sum']:<9.3f}({s['fr-mean']:.3f})  "
    logstr+= f"wattr:{s['wfa-sum']:<9.3f}({s['wfa-mean']:.3f})  "
    logstr+= f"wrepl:{s['wfr-sum']:<9.3f}({s['wfr-mean']:.3f})  "
    logstr+= f"sum-all:{s['f-all']:<9.3f}  "
    logstr+= f"relocs:{s['relocs-sum']:<9.3f}({s['relocs-mean']:.3f})  "
    logstr+= f"weighted-relocs:{s['wrelocs-sum']:<9.3f}({s['wrelocs-mean']:.3f})  "

    return s, logstr

class StatsLog (Callback_Base):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.statsdf = pd.DataFrame()
        self.log = ReportLog(args.logfilepath)
    
        self.emb_filepath = f"{args.outputdir}/{args.outputfilename}" # the path to store the latest embedding
        self.hist_filepath = f"{args.outputdir}/{args.historyfilename}" # the path to APPEND the latest embedding
        self.stats_filepath = f"{args.outputdir}/{args.statsfilename}" # the path to save the latest stats

        os.makedirs(args.outputdir, exist_ok=True)

    def save_embeddings(self, fd_model, **kwargs):
        emb = fd_model.get_embeddings()
        # save embeddings as pandas df
        df = pd.DataFrame(emb, index=fd_model.Gx.nodes())
        df.to_pickle(self.emb_filepath)
    
    def save_history(self, fd_model, **kwargs):
        emb = fd_model.get_embeddings()
        # save embeddings history
        with open(self.hist_filepath, "ab") as f: # append embeddings
            pickle.dump(emb, f)

    def update_stats(self, fd_model, epoch, **kwargs):
        # make stats and logs
        stats, logstr = make_stats_log(fd_model, epoch)
        self.log.print(logstr)

        # make the stats as dataframe
        if(len(self.statsdf) == 0):  # new dataframe
            self.statsdf = pd.DataFrame(columns=list(stats.keys()))    
        
        stats = pd.DataFrame(stats, index=[0])

        self.statsdf = pd.concat([self.statsdf, stats], ignore_index=True)

        # Save DataFrame to a CSV file
        temp_filename = self.stats_filepath+'.tmp'
        self.statsdf.to_csv(temp_filename, index=False)
        # Rename the temporary file to the final filename
        os.rename(temp_filename, self.stats_filepath)

    def on_epoch_end(self, fd_model, epoch, **kwargs):
        self.update_stats(fd_model, epoch, **kwargs)
        self.save_embeddings(fd_model, **kwargs)
        if(epoch % args.save_history_every == 0):
            self.save_history(fd_model, **kwargs)
            
            
    def on_train_end(self, fd_model, epochs, **kwargs):
        print("Final save")
        self.save_embeddings(fd_model, **kwargs)
        self.save_history(fd_model, **kwargs)
        
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


def openfile(path, mode):
    try:
        return open(path, mode)
    except Exception as e:
        print(f"Failed to open the file at {path}")
        print(e)
        return None

def remove_comment(line, cmt='#'):
    idx = line.find(cmt)
    if(idx == -1):      return line
    elif(idx == 0):     return ''
    else:               return line[:idx]

def read_labels(filepath, skip_head=False, multi_label=False):
    Y = dict()
    fin = openfile(filepath, 'r')
    if skip_head:
        fin.readline()
    for i, line in enumerate(fin.readlines(), start=1+int(skip_head)):
        line = remove_comment(line)
        if len(line) == 0:
            continue
        vec = line.strip().split(' ')
        if (len(vec) == 1 ):
            raise ValueError(f"Label file {filepath} has invalid format at line {i+1}")
        if(multi_label is False):
            Y[vec[0]] = vec[1]
        else:
            Y[vec[0]] = [vec[i] for i in range(1, len(vec))]
    fin.close()
    return Y

def process_arguments(
                # You can override the following default parameters by argument passing
                EMBEDDING_METHOD='forcedirected',
                DEFAULT_DATASET='ego-facebook',
                DATASET_CHOICES=['tinygraph', 'cora', 'citeseer', 'pubmed', 'ego-facebook', 'corafull', 'wiki'],
                NDIM=12, ALPHA=0.3,
                ):
    
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-d', '--dataset', type=str, default=DEFAULT_DATASET, choices=DATASET_CHOICES, 
                        help='name of the dataset (default: cora)')
    parser.add_argument('--outputdir', type=str, default=None, 
                        help=f"output directory (default: ./embeddings-tmp/{EMBEDDING_METHOD}_v{FDModel.VERSION}_{{ndim}}d/{DEFAULT_DATASET})")
    parser.add_argument('--outputfilename', type=str, default='embed-df.pkl', 
                        help='filename to save the final result (default: embed-df.pkl). Use pandas to open.')
    parser.add_argument('--historyfilename', type=str, default='embed-hist.pkl', 
                        help='filename to store s sequence of results from each iteration (default: embed-hist.pkl). Use pickle loader to open.')
    parser.add_argument('--save-history-every', type=int, default=5, 
                        help='save history every n iteration.')
    parser.add_argument('--statsfilename', type=str, default='stats.csv', 
                        help='filename to save the embedding stats history (default: stats.csv)')
    parser.add_argument('--logfilepath', type=str, default=None, 
                        help='path to the log file (default: {EMBEDDING_METHOD}-{dataset}.log)')

    parser.add_argument('--ndim', type=int, default=NDIM, 
                        help='number of embedding dimensions (default: 12)')
    parser.add_argument('--alpha', type=float, default=ALPHA,
                        help='alpha parameter (default: 0.3)')

    parser.add_argument('--epochs', type=int, default=4000, 
                        help='number of iterations for embedding process (default: 4000)')

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
        
    # use default parameter values if required

    if(args.outputdir is None):
        args.outputdir = f"./embeddings-tmp/{EMBEDDING_METHOD}_v{FDModel.VERSION}_{args.ndim}d/{args.dataset}"
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
    
    DATA_ROOT=os.path.expanduser('~/gnn/datasets')       # root of datasets to obtain graphs from
    # EMBEDS_ROOT=os.path.expanduser('~/gnn/embeddings')    # root of embeddings to save to
    if(args.dataset in ['cora', 'pubmed', 'citeseer', 'tinygraph', 'ego-facebook', 'corafull']):
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
        args.edgelist = f'{DATA_ROOT}/BlogCatalog-dataset/data/edge.csv'
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

    ##### START EMBEDDING #####    
    model = FDModel(Gx, n_dims=args.ndim, alpha=args.alpha, callbacks=[StatsLog(), EarlyStopping()])
    model.train(epochs=args.epochs, device=device)
    
    ##### SAVE EMBEDDINGS #####
    embeddings = model.get_embeddings()
    

    pos_dict = dict(zip(Gx.nodes(), model.Z[:,:2].cpu().numpy()))
    nx.draw(Gx, pos=pos_dict, node_size=10, width=0.1, node_color='black', edge_color='gray')

# %%
