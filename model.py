# %%
# # reload modules
# %load_ext autoreload
# %autoreload 2

# get arguments
import argparse
import os
import pickle
from typing import Any
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx, networkit # this line is for typing hints used in this code
import networkx as nx
import networkit as nk
from types import *

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


class ForceDirected(Model_Base):
    def __init__(self, Gx, n_dims, alpha:float,
                random_points_generator:callable = generate_random_points, 
                **kwargs):
        super().__init__(**kwargs)
        self.Gx = Gx
        self.n_nodes = Gx.number_of_nodes()
        self.n_dims = n_dims
        self.alpha = alpha
        Gnk, A, degrees, hops = process_graph_networkx(Gx)

        self.hops = hops
        self.degrees = degrees

        # find max hops
        self.maxhops = max(hops[hops<=self.n_nodes]) # hops<=n to exclude infinity values
        hops[hops>self.maxhops]=self.maxhops+1  # disconncted nodes are 'maxhops+1' hops away, to avoid infinity
        print("max hops:", self.maxhops)
        
        self.alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=self.hops, alpha=self.alpha)
        
        self.fmodel_attr = ForceClass(name='attractive', 
                            func=lambda fd_model: attractive_force_ahops(fd_model.D, fd_model.N, fd_model.unitD, fd_model.alpha_hops, k1=1, k2=1)
                            )
        self.fmodel_repl = ForceClass(name='repulsive',
                            #  func=lambda fd_model: repulsive_force_hops_exp(fd_model.D, fd_model.N, fd_model.unitD, torch.tensor(fd_model.hops).to(fd_model.device), k1=10, k2=0.9)
                            func=lambda fd_model: repulsive_force_hops_exp(fd_model.D, fd_model.N, fd_model.unitD, fd_model.hops, k1=10, k2=0.9)
                            )
        # To be used like the following
        # result_F = self.fmodel_attr(self) # pass the current model (with its contained embeddings) to calculate the force
        
        # self.random_drop = DropLinearChange(name='linear-increase', start=0.1, end=0.9, change_rate=0.001)
        self.random_drop = DropSteadyRate(name='steady-rate', drop_rate=0.5)
        
        self.statsdf = pd.DataFrame()
        # initialize embeddings
        self.random_points_generator = random_points_generator
        self.embeddings = NodeEmbeddingClass(self.n_nodes, self.n_dims)
        self.embeddings.set(self.random_points_generator(self.n_nodes, self.n_dims))
        pass
    
    @property
    def Z(self):        return self.embeddings.Z
    @property
    def D(self):        return self.embeddings.D
    @property
    def N(self):        return self.embeddings.N
    @property
    def unitD(self):    return self.embeddings.unitD

    def get_embeddings(self):   return self.embeddings.Z.detach().cpu().numpy()

    def train(self, epochs=100, device=None, **kwargs):
        # train begin
        kwargs['epochs'] = epochs
        self.notify_train_begin_callbacks(**kwargs)

        # initialize train
        self.device = device
        if(device is None):
            self.device = torch.device('cpu')
        
        
        self.hops = torch.tensor(self.hops).to(device)
        self.alpha_hops = torch.tensor(self.alpha_hops).to(device)
        self.degrees = torch.tensor(self.degrees).to(device) # mass of a node is equivalent to its degree
        self.embeddings.to(device)

        self.dZ = torch.zeros_like(self.Z).to(device)
        
        for epoch in range(epochs):
            # epoch begin
            kwargs['epoch'] = epoch
            self.notify_epoch_begin_callbacks(**kwargs)
            self.notify_batch_begin_callbacks(**kwargs)
            
            print(f'Epoch {epoch+1}/{epochs}')
            ###################################
            # this is the forward pass
            Fa = self.fmodel_attr(self) # pass the current model (with its contained embeddings) to calculate the force
            force = torch.norm(Fa, dim=1)
            fsum, fmean, fstd = torch.sum(force), torch.mean(force), torch.std(force)
            # print(f'attr: {fsum:,.3f}({fmean:.3f})  ', end='')

            Fr = self.fmodel_repl(self)
            force = torch.norm(Fr, dim=1)
            fsum, fmean, fstd = torch.sum(force), torch.mean(force), torch.std(force)
            # print(f'repl: {fsum:,.3f}({fmean:.3f})  ' , end='')
            Fsum = Fa+Fr
            force = torch.norm(Fsum, dim=1)
            fsum, fmean, fstd = torch.sum(force), torch.mean(force), torch.std(force)
            # print(f'sum: {fsum:,.3f}({fmean:.3f})  ' , end='')
            F = self.random_drop(Fa+Fr, **kwargs)
            
            ###################################
            # finally calculate the gradient and udpate the embeddings
            # find acceleration on each point a = F/m. And X-X0 = a*t^2, Assume X0 = 0 and t = 1
            self.dZ = torch.where(self.degrees[:, None] != 0, F / self.degrees[:, None], torch.zeros_like(F))
            relocs = torch.norm(self.dZ, dim=1)
            # print(f'relocs: {relocs.sum():,.3f}({relocs.mean():.5f})')
            self.embeddings.update(self.dZ)
            ###################################

            # batch ends
            self.notify_batch_end_callbacks(**kwargs)
            self.notify_epoch_end_callbacks(**kwargs)
        # epoch ends            
        
        self.notify_train_end_callbacks(**kwargs)
        pass

def make_hops_stats(N, hops, maxhops):
    """N is pairwise euclidean distance, numpy/torch tensor
    hops is pairwise hop distance, numpy/torch tensor
    N and hops must have the same shape
    """
    if(isinstance(N, torch.Tensor)):
        _mean = lambda x: x.mean().item()
        _std  = lambda x: x.std().item()
        _max  = lambda x: x.max().item()
    else:
        _mean = lambda x: x.mean()
        _std  = lambda x: x.std()
        _max  = lambda x: x.max()
        
    s = {}
    maxhops = int(maxhops)
    # connected components
    for h in range(1, maxhops+1):
        tN = N[ hops==h ]
        if( len( tN )>0 ):
            s[f"hops{h}_mean"] = _mean( tN )
            s[f"hops{h}_std"]  = _std( tN )
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
        s[f"hopsinf_mean"] = _mean( tN )
        s[f"hopsinf_std"]  = _std( tN )
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
                DATASET_CHOICES=['tinygraph', 'cora', 'citeseer', 'pubmed', 'ego-facebook', 'corafull', 'wiki']
                ):
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-d', '--dataset', type=str, default=DEFAULT_DATASET, choices=DATASET_CHOICES, 
                        help='name of the dataset (default: cora)')
    parser.add_argument('--outputdir', type=str, default=None, 
                        help=f"output directory (default: ./embeddings-tmp/{EMBEDDING_METHOD}/{DEFAULT_DATASET})")
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

    parser.add_argument('--ndim', type=int, default=12, 
                        help='number of embedding dimensions (default: 12)')
    parser.add_argument('--alpha', type=float, default=0.3,
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
        args.outputdir = f"./embeddings-tmp/{EMBEDDING_METHOD}/{args.dataset}"
    if(args.logfilepath is None):
        args.logfilepath = f"./{args.outputdir}/{EMBEDDING_METHOD}-{args.dataset}.log"

    # Combine the description words into a single string
    args.description = ' '.join(args.description)

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
    
    DATA_ROOT='~/gnn/datasets'       # root of datasets to obtain graphs from
    EMBEDS_ROOT='~/gnn/embeddings'    # root of embeddings to save to
    if(args.dataset in ['cora', 'pubmed', 'citeseer', 'tinygraph', 'ego-facebook', 'corafull']):
        args.path_nodes = f'{DATA_ROOT}/{args.dataset}/{args.dataset}.nodes'
        args.path_features = f'{DATA_ROOT}/{args.dataset}/{args.dataset}.x'
        args.path_labels = f'{DATA_ROOT}/{args.dataset}/{args.dataset}.y'
        args.path_edgelist = f'{DATA_ROOT}/{args.dataset}/{args.dataset}.edgelist'
    elif(args.dataset == 'wiki'):
        args.path_nodes = f'{DATA_ROOT}/wiki/Wiki_nodes.txt'
        args.path_features = f'{DATA_ROOT}/wiki/Wiki_category.txt'
        args.path_labels = f'{DATA_ROOT}/wiki/Wiki_labels.txt'
        args.path_edgelist = f'{DATA_ROOT}/wiki/Wiki_edgelist.txt'
    elif(args.dataset == 'blogcatalog'):
        args.path_nodes = f'{DATA_ROOT}/BlogCatalog-dataset/data/nodes.csv'
        args.path_edgelist = f'{DATA_ROOT}/BlogCatalog-dataset/data/edge.csv'
        args.path_labels = f'{DATA_ROOT}/BlogCatalog-dataset/data/labels.csv' # the first column is the node id, the second column is the label


    args.embedding = f'{EMBEDS_ROOT}/forcedirected/{args.dataset}/embed.npy'

    def graph_loader(paths: dict()):
        ##### LOAD GRAPH #####
        Gx = nx.Graph()
        # load nodes first to keep the order of nodes as found in nodes or labels file
        if(os.path.exists(args.path_nodes)):
            Gx.add_nodes_from(np.loadtxt(args.path_nodes, dtype=str, usecols=0))
            print('loaded nodes from nodes file')
        elif(os.path.exists(args.path_features)):
            Gx.add_nodes_from(np.loadtxt(args.path_features, dtype=str, usecols=0))
            print('loaded nodes from features file')
        elif(os.path.exists(args.path_labels)):
            Gx.add_nodes_from(np.loadtxt(args.path_labels, dtype=str, usecols=0))   
            print('loaded nodes from labels file')

        # add edges from args.path_edgelist
        Gx.add_edges_from(np.loadtxt(args.path_edgelist, dtype=str))
    Gx = graph_loader(args.__dict__)

    print("Number of nodes:", Gx.number_of_nodes())
    print("Number of edges:", Gx.number_of_edges())
    print("Number of connected components:", nx.number_connected_components(Gx))    

    # Y = read_labels(args.path_labels)

    if(args.device == 'auto'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    ##### START EMBEDDING #####    
    model = ForceDirected(Gx, n_dims=args.ndim, alpha=args.alpha, callbacks=[StatsLog()])
    model.train(epochs=args.epochs, device=device)
    
    ##### SAVE EMBEDDINGS #####
    embeddings = model.get_embeddings()
    

    pos_dict = dict(zip(Gx.nodes(), model.Z[:,:2].cpu().numpy()))
    nx.draw(Gx, pos=pos_dict, node_size=10, width=0.1, node_color='black', edge_color='gray')

# %%
