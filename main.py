
# %%
# reload modules
# %load_ext autoreload
# %autoreload 2

# get arguments
import os, sys
import argparse
import pickle
from typing import Any

import logging
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
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
ch.setFormatter(logging.Formatter('%(message)s'))
log.addHandler(ch)

# %%

from model_1 import FDModel as FDModel_1
from model_2 import FDModel as FDModel_2
from model_3 import FDModel as FDModel_3
from model_4 import FDModel as FDModel_4
from model_4nodrop import FDModel as FDModel_4nodrop

def make_hops_stats(Z, hops, maxhops):
    print("make_hops_stats")
    # print("make_hops_stats", Z.device, f"block_size:{block_size}", f"ndim:{Z.size(1)}", f"block bytesize:{Z.element_size()*block_size*Z.size(1):_d}")
    maxhops = int(maxhops)
    sum_x=torch.zeros(maxhops+2).to(Z.device)
    sum_x2=torch.zeros(maxhops+2).to(Z.device)

    i,j = torch.triu_indices(Z.size(0), Z.size(0), offset=1) # offset 1 to exclude self distance
    """Z is node embedding, torch tensor"""
    # empty cuda cache
    torch.cuda.empty_cache()
    # get available cuda memory
    available_memory = torch.cuda.get_device_properties(Z.device).total_memory - torch.cuda.memory_allocated(Z.device)
    block_size = int(available_memory*0.9) // (Z.element_size()*Z.size(1)) # 10GB
    print(f"available_memory {available_memory//2**30} GB")
    print(f"block_size {block_size} elements")
    print(f"block_size {block_size*Z.element_size()*Z.size(1)/2**30} GB")
        
    for b in range(0, len(i), block_size):
        print(b, b+block_size)
        i_block = i[b:b+block_size]
        j_block = j[b:b+block_size]
        # print(f"block={b}", f"i_block:{i_block[0]}-{i_block[-1]}", f"j_block:{j_block[0]}-{j_block[-1]}")
        hops_block = hops[i_block, j_block]
        N_block = torch.norm(Z[j_block] - Z[i_block], dim=-1)

        for h in range(1, maxhops+1):
            hmask = hops_block==h
            sum_x[h] += N_block[hmask].sum()
            sum_x2[h] += (N_block[hmask]**2).sum()
        # disconnected components
        hmask = hops_block>maxhops 
        sum_x[maxhops+1] += N_block[hmask].sum()
        sum_x2[maxhops+1] += (N_block[hmask]**2).sum()
        del hops_block, N_block
    print("make_hops_stats done")
    s={}
    s.update({f"hops{h}_mean": (sum_x[h]/(hops==h).sum()).item() for h in range(1, maxhops+1)})
    s.update({f"hops{h}_std":  (sum_x2[h]/(hops==h).sum() - (sum_x[h]/(hops==h).sum())**2).sqrt().item() for h in range(1, maxhops+1)})
    # disconnected components
    if((hops>maxhops).sum()>0):
        s[f"hops{maxhops}_mean"] = (sum_x[maxhops]/(hops>maxhops).sum()).item()
        s[f"hops{maxhops}_std"]  = (sum_x2[maxhops]/(hops>maxhops).sum() - (sum_x[maxhops]/(hops>maxhops).sum())**2).sqrt().item()
    return s

def make_stats_log(model, epoch):
    logstr = ''
    s = {'epoch': epoch}

    s.update(make_hops_stats(model.Z, model.hops, model.maxhops))

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

    logstr = f'{epoch}| '
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
        self.emb_filepath = f"{args.outputdir}/{args.outputfilename}" # the path to store the latest embedding
        self.hist_filepath = f"{args.outputdir}/{args.historyfilename}" # the path to APPEND the latest embedding
        self.stats_filepath = f"{args.outputdir}/{args.statsfilename}" # the path to save the latest stats

        self.statsdf = pd.DataFrame()
        # self.logger = ReportLog(args.logfilepath)
        
        # make he stat logger #######################
        self.statlog = logging.getLogger('StatsLog')
        self.statlog.setLevel(logging.INFO)

        # Add a file handler to write logs to a file
        file_handler = logging.FileHandler(args.logfilepath)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s: %(message)s'))
        self.statlog.addHandler(file_handler)

        # Add a console handler to print logs to the console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter('> %(message)s'))
        self.statlog.addHandler(console_handler)
        ############################################
    

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
        self.statlog.info(logstr) # FIX THIS

        # make the stats as dataframe
        if(len(self.statsdf) == 0):  # new dataframe
            self.statsdf = pd.DataFrame(columns=list(stats.keys()))    
        
        stats = pd.DataFrame(stats, index=[0])

        self.statsdf = pd.concat([self.statsdf, stats], ignore_index=True)

        # Save DataFrame to a CSV file
        # temp_filename = self.stats_filepath+'_tmp'
        self.statsdf.to_csv(self.stats_filepath, index=False)
        # Rename the temporary file to the final filename
        # os.rename(temp_filename, self.stats_filepath)

    def on_epoch_begin(self, fd_model, epoch, epochs, **kwargs):
        log.info(f'Epoch {epoch+1}/{epochs}')
        # return super().on_batch_begin(fd_model, epoch, **kwargs)

    def on_epoch_end(self, fd_model, epoch, **kwargs):
        self.update_stats(fd_model, epoch, **kwargs)
        self.save_embeddings(fd_model, **kwargs)
        if(epoch % args.save_history_every == 0):
            self.save_history(fd_model, **kwargs)
            
    def on_batch_end(self, fd_model, batch, **kwargs):
        log.info(f"   Batch {batch}/{kwargs['batches']} ({kwargs['row_batch_size']}/{kwargs['max_batch_size']})")
        # return super().on_batch_begin(fd_model, batch, **kwargs)

    def on_train_end(self, fd_model, epochs, **kwargs):
        log.info("Final save")
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
                OUTPUTDIR_ROOT='./embeddings-tmp',
                DATASET_CHOICES=['tinygraph', 'cora', 'citeseer', 'pubmed', 'ego-facebook', 'corafull', 'wiki', 'blogcatalog', 'flickr', 'youtube'],
                NDIM=128, ALPHA=0.3,
                ):
    
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-d', '--dataset', type=str, default=DEFAULT_DATASET, choices=DATASET_CHOICES, 
                        help='name of the dataset (default: cora)')
    parser.add_argument('-v', '--fdversion', type=str, default='4', choices=['1','2','3','4','4nodrop'],
                        help='version of the force-directed model (default: 4)')
    parser.add_argument('--outputdir', type=str, default=None, 
                        help=f"output directory (default: {OUTPUTDIR_ROOT}/{EMBEDDING_METHOD}_v{{version}}_{{ndim}}d/{DEFAULT_DATASET})")
    parser.add_argument('--outputfilename', type=str, default='embed-df.pkl', 
                        help='filename to save the final result (default: embed-df.pkl). Use pandas to open.')
    parser.add_argument('--historyfilename', type=str, default='embed-hist.pkl', 
                        help='filename to store s sequence of results from each iteration (default: embed-hist.pkl). Use pickle loader to open.')
    parser.add_argument('--save-history-every', type=int, default=100, 
                        help='save history every n iteration.')
    parser.add_argument('--statsfilename', type=str, default='stats.csv', 
                        help='filename to save the embedding stats history (default: stats.csv)')
    parser.add_argument('--logfilepath', type=str, default=None, 
                        help='path to the log file (default: {EMBEDDING_METHOD}-{dataset}.log)')

    parser.add_argument('--ndim', type=int, default=NDIM, 
                        help=f'number of embedding dimensions (default: {NDIM})')
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
    if  (args.fdversion=='1'): args.FDModel = FDModel_1
    elif(args.fdversion=='2'): args.FDModel = FDModel_2
    elif(args.fdversion=='3'): args.FDModel = FDModel_3
    elif(args.fdversion=='4'): args.FDModel = FDModel_4
    elif(args.fdversion=='4nodrop'): args.FDModel = FDModel_4nodrop
    
    if(args.outputdir is None):
        args.outputdir = f"{OUTPUTDIR_ROOT}/{EMBEDDING_METHOD}_v{args.FDModel.VERSION}_{args.ndim}d/{args.dataset}"
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

    ##### START EMBEDDING #####    
    model = args.FDModel(Gx, n_dims=args.ndim, alpha=args.alpha, callbacks=[StatsLog(), EarlyStopping()])
    model.train(epochs=args.epochs, device=device)
    
    ##### SAVE EMBEDDINGS #####
    embeddings = model.get_embeddings()
    

    pos_dict = dict(zip(Gx.nodes(), model.Z[:,:2].cpu().numpy()))
    nx.draw(Gx, pos=pos_dict, node_size=10, width=0.1, node_color='black', edge_color='gray')

# %%
