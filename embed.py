# %%
# get arguments
import argparse
from typing import Any
from pprint import pprint

DEFAULT_DATASET='cora'
# DEFAULT_DATASET='ego-facebook'
DATASET_CHOICES=['cora', 'citeseer', 'pubmed', 'ego-facebook']
parser = argparse.ArgumentParser(description='Process command line arguments.')
parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET, choices=DATASET_CHOICES, 
                    help='name of the dataset (default: cora)')
parser.add_argument('--outputdir', type=str, default=None, 
                    help=f"output directory (default: ./embeddings-tmp/{DEFAULT_DATASET})")
parser.add_argument('--outputfilename', type=str, default='embed.npy', 
                    help='filename to save the final result (default: embed.npy). Use numpy.load() to open.')
parser.add_argument('--historyfilename', type=str, default='embed-hist.pkl', 
                    help='filename to store s sequence of results from each iteration (default: embed-hist.pkl). Use pickle loader to open.')
parser.add_argument('--statsfilename', type=str, default='stats.csv', 
                    help='filename to save the embedding stats history (default: stats.csv)')
parser.add_argument('--logfilepath', type=str, default=None, 
                    help='path to the log file (default: {dataset}-log.txt)')
parser.add_argument('--ndim', type=int, default=12, 
                    help='number of embedding dimensions (default: 12)')
parser.add_argument('--epochs', type=int, default=4000, 
                    help='number of iterations for embedding process (default: 4000)')
parser.add_argument('--alpha', type=float, default=0.3,
                    help='alpha parameter (default: 0.3)')
parser.add_argument('--std0', type=float, default=1.0, 
                    help='initialization standard deviation (default: 1.0).')
parser.add_argument('--base_std0', type=float, default=0.005, 
                    help='base standard deviation of noise (default: 0.005), noise_std = f(t)*std0 + base_std0. f(t) converges to 0.')
parser.add_argument('--add-noise', action='store_true', 
                    help='enable noise')

parser.add_argument('--random-drop', default='steady-rate', type=str,
                    choices=['steady-rate', 'exponential-rate', 'linear-rate', 'none', ''], 
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
    args.outputdir = f"./embeddings-tmp/{args.dataset}"
if(args.logfilepath is None):
    args.logfilepath = f"./{args.outputdir}/{args.dataset}-log.txt"

# Combine the description words into a single string
args.description = ' '.join(args.description)

print("\nArguments:")
for key, value in vars(args).items():
    print(f"{key:20}: {value}")
print("")

# %%
# %reload_ext autoreload
# %autoreload 2
import os
import concurrent.futures
import time

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)

import networkit as nk
import networkx as nx

import torch

from sklearn.metrics.pairwise import euclidean_distances
import sklearn as sk
# %%
class DummyType:
    def __init__(self, _name, _value=None) -> None:
        self._name = _name
        self._value = _value
        pass

    @property
    def name(self):
        return self._name
    
    @property
    def value(self):
        return self._value

    def __repr__(self) -> str:
        return str(self._value)+"("+self._name+":"+str(type(self._value))+")"

    def __iadd__ (self, other):
        # print("DummyType iadd", self, other)
        if(self._value is None):
            self._value = other
        else:
            self._value += other

class DummyClass:
    def __init__(self) -> None:
        pass

    def __setattr__(self, attr, value):
        # if(attr in self.__dict__):
        #     print("DummyClass setattr", attr, self.__dict__[attr], "->", value)
        # else:
        #     print("DummyClass setattr", attr, "->", value)
        if(value is not None):
            self.__dict__[attr] = value
            # print("    set value to", value)
            return

        if(attr not in self.__dict__):
            self.__dict__[attr] = DummyType(attr)
            return
        
        
        if(type(self.__dict__[attr]) is not DummyType):
            self.__dict__[attr] = value
        elif(type(self.__dict__[attr]) is DummyType):
            self.__dict__[attr] = self.__dict__[attr].value
            # print("    change type from DummyType to", type(self.__dict__[attr]))
        

    def __getattr__(self, attr):
        # print("DummyClass getattr", attr)
        if(attr not in self.__dict__):
            # print("    create", attr)
            self.__dict__[attr] = DummyType(attr)
        elif(type(self.__dict__[attr]) is DummyType):
            if(self.__dict__[attr].value is not None):
                self.__dict__[attr] = self.__dict__[attr].value

        return self.__dict__[attr]
    
# Create an instance of the class
durations = DummyClass()
print(durations.testval)

# Access attributes and perform operations
someval = 7
durations.testval += someval
print(durations.testval)  # Output: 7

# %%
############################################################################
# Load data
############################################################################
# Get the Cora dataset

from utilities import load_graph_networkx, process_graph_networkx

Gx, data = load_graph_networkx(datasetname=args.dataset, rootpath='../datasets')
G, A, degrees, hops = process_graph_networkx(Gx)
n = A.shape[0]
# find max hops
maxhops = max(hops[hops<n+1]) # hops<n+1 to exclude infinity values
hops[hops>maxhops]=maxhops+1
print("max hops:", maxhops)
    

# %%
############################################################################
# GENERATE RANDOM POINTS
############################################################################

# randomly distribute the initial position of nodes in a normal distribution with
def generate_random_samples(n, d, s):
    # Generate random points from a standard normal distribution
    P = np.random.normal(0, 1, size=(n, d))
    # Normalize the points to have unit norm
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    P = P / norms
    # Scale the points to have the desired standard deviation
    distances = np.random.normal(0, s, size=(n, 1))
    P *= distances
    return P

n = G.numberOfNodes() # number of nodes
ndim = args.ndim
std0=args.std0

Z = generate_random_samples(n, ndim, std0)
print(f"A matrix of shape {Z.shape} where each row corresponds to embedding of a node in a {ndim}-dimensional space.")
# %%
def make_embedding_stats(Nt, hops, maxhops):
    """Nt is pairwise euclidean distance, Numpy
    hops is pairwise hop distance, Numpy
    """
    ht = hops[hops<=maxhops]
    
    stats = {}
    for l in range(1, int(ht.max()+1)):
        mask = hops==l
        if(len(Nt[mask])>0):
            stats[f"hops{l}_mean"]=Nt[mask].mean()
            stats[f"hops{l}_std"]=Nt[mask].std()
        # print(f"{l:3d} {Nt[mask].mean():10.3f} {Nt[mask].std():10.3f} {len(Nt[mask])/2:8.0f}")
    
    # disconnected components
    mask = hops>maxhops
    if(len(Nt[mask])>0):
        stats[f"hopsinf_mean"]=Nt[mask].mean()
        stats[f"hopsinf_std"]=Nt[mask].std()
    return stats
    # print(f"inf {Nt[mask].mean():10.3f} {Nt[mask].std():10.3f} {len(Nt[mask])/2:8.0f}")

import nodeforce as nf


## random drop class
class RandomDrop:
    def __init__(self, drop_mode:str, drop_params) -> None:
        self.drop_mode = drop_mode
        self.drop_params = drop_params
        self.t = 0
        self._indices = None
        self._initialize()
        pass

    def __call__(self, X, shape=None, t=None):
        self.drop(X, shape, t)
        pass
    
    def drop(self, X, shape=None, t=None, use_latest=False):
        if(use_latest): # don't generate new indieces
            X[self._indices] = 0
            return X    
        if(t is None):
            t = self.t
            self.t += 1
        if(shape is None):
            shape = X.shape
        self._indices = self.generate_indices(shape, t)
        X[self._indices] = 0
        return X
    
    @property
    def enabled(self):
        return (self.drop_mode not in ['', 'none', 'None'])

    @property
    def last_indices(self):
        self._indices
    
    def _initialize(self):
        drop_params = self.drop_params
        if(self.drop_mode == 'steady-rate'):
            self.drop_rate = 0.5 # default value
            if(len(drop_params)>0):
                self.drop_rate = drop_params[0]
            self.generate_indices = lambda shape, *args: torch.rand(shape) > self.drop_rate

        elif(self.drop_mode == 'linear-rate'):
            self.r_grad = 0.0003 # will end after 3333 iterations
            self.r_start = 1.0
            self.r_end = 0.0
            if(len(drop_params)>0):
                self.r_grad = drop_params[0]
            if(len(drop_params)>1):
                self.r_start = drop_params[1]
                self.r_end = drop_params[2]
            
            self.drop_rate = self.r_start
            def _dummyfunc(shape, t=0, *args):
                self.drop_rate = self.r_start - t*self.r_grad
                if (self.drop_rate < self.r_end): 
                    self.drop_rate = self.r_end
                idx = torch.rand(shape) < self.drop_rate
                return idx
                
            self.generate_indices = _dummyfunc
        
        elif(self.drop_mode == 'exponential-rate'):
            self.r_amp = 1.0
            self.r_k = 1.0                
            if(len(drop_params)>0):
                self.r_amp = drop_params[0]
                self.r_k = drop_params[1]
            
            self.drop_rate = self.r_amp
            def _dummyfunc(shape, t=0, *args):
                self.drop_rate = self.r_amp * np.exp(-t*self.r_k)
                idx = torch.rand(shape) < self.drop_rate
                return idx
            self.generate_indices = _dummyfunc
        
        else: #don't drop any
            self.generate_indices = lambda shape, *args: torch.zeros(shape, dtype=torch.bool)

random_drop = RandomDrop(drop_mode=args.random_drop, drop_params=args.random_drop_params)


def test(
    # self,
    train_z: torch.Tensor, train_y: torch.Tensor,
    test_z: torch.Tensor,  test_y: torch.Tensor,
    solver: str = 'lbfgs', multi_class: str = 'auto', *args, **kwargs,
) -> float:
    r"""Evaluates latent space quality via a logistic regression downstream
    task."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                **kwargs).fit(train_z.detach().cpu().numpy(),
                                            train_y.detach().cpu().numpy())
    return clf.score(test_z.detach().cpu().numpy(),
                     test_y.detach().cpu().numpy())


############################################################################
# Embed
############################################################################
def embed_forcedirected(Z, degrees, hops):
    statsdf = pd.DataFrame() # dataframe for embedding statistics
    
    # FILES AND PATHS
    # Get the absolute path of the directory
    abspath = os.path.abspath(args.outputdir)
    os.makedirs(abspath, exist_ok=True)

    ofilepath = f"{args.outputdir}/{args.outputfilename}" # the path to store the latest embedding
    histfilepath = f"{args.outputdir}/{args.historyfilename}" # the path to APPEND the latest embedding
    # Reset the binary files if any already exists
    for fpath in [ofilepath, histfilepath]:
        if os.path.exists(fpath):
            os.remove(fpath)
    
    # TORCH DEVICES
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    # device_str = "cpu"
    device = torch.device(device_str)
    # torch.cuda.set_device(0)
    print("Device type:", device_str, device)
    
    # LOG ALL ARGUMENTS
    logstr="Arguments:\n"
    for key, value in vars(args).items():
        logstr+=f"{key:20}: {value}\n"
    
    logstr+=f"\ndevice type: {device_str}\n"
    with open(args.logfilepath, "w") as f: # write to the logfile
        f.write(logstr)

    alpha = args.alpha

    # standard deviation of noise = f(t)*std_0 + base_std0, 
    # where f(t) is a damping function converging to 0
    std0 = args.std0
    base_std0 = args.base_std0

    # BURN-IN PHASE, experimental
    burnin_iterations = 0
    burnin_steady_rate = False
    burnin_rate = 0.05
    if (burnin_iterations>0): burnin_rate = 1.0/burnin_iterations
    max_rate = 1
    rate = 1

    
    # alpha_hops = get_alpha_hops(hops, alpha)
    alpha_hops = np.apply_along_axis(nf.get_alpha_hops, axis=1, arr=hops, alpha=alpha)
    alpha_hops = torch.tensor (alpha_hops).to(device)
    
    mass = torch.tensor(degrees).to(device) # mass of a node is equivalent to its degree
    Z = torch.tensor(Z).to(device)
    Z_0 = Z.detach().clone().to(device)
    V0 = torch.zeros_like(Z).to(device)
    Fa = torch.zeros_like(Z).to(device)
    Fr = torch.zeros_like(Z).to(device)

    
    durations = DummyClass() # for timer and counter
    durations.start = time.time()
    durations.elapsed = lambda:time.time() - durations.start
    
    stderr_change = 0
    max_iterations = args.epochs
    for _iter in range(max_iterations): # _iter MUST START FROM 0
        
        logstr = f"iter{_iter:4d} | "
        
        # rate calculation for burnin phase 
        if(_iter >= burnin_iterations):
            rate = max_rate
        else:
            if (burnin_steady_rate == True):
                rate = burnin_rate
            else:
                rate += burnin_rate
                
            s = f"/{burnin_iterations}, rate:{rate:.2f}"
            print(s)
            logstr += f"{s} "

        durations.iteration += 1

        # randomly select values (on any axis) for change and modification
        random_select = 1 # default is neutral value
        if(random_drop.enabled):
            random_select = random_drop.generate_indices(Z.shape, _iter)
            random_select = random_select.float().to(device)

        # Add noise if enabled. Noise level is diminishing over time. 
        std_noise = 0 # defaults are is neutral values
        relocation_noise = 0 
        if(args.add_noise):
            # std_noise = np.exp(-stderr_change)*std0 + np.exp(-stderr_change/100)*std0*0.01 
            std_noise = np.exp(-stderr_change)*std0 + base_std0
            # std_noise = np.exp(-stderr_change)*std0 
            std_noise = torch.tensor(std_noise).to(device)
            stderr_change += 0.005
            relocation_noise = torch.normal(0, std_noise, size=Z.shape, device=device)*random_select
        
        t = time.time()
        # Calculate forces
        D = nf.pairwise_difference(Z) # D[i,j] = Z[i] - Z[j]
        durations.t_pairwisediff = time.time() - t
        t = time.time()
        N = torch.norm(D, dim=-1)     # pairwise distance between points
        durations.t_pairwisenorm = time.time() - t
        # Element-wise division with mask
        mask = N!=0 
        unitD = torch.zeros_like(D)   # unit direction
        unitD[mask] = D[mask] / N[mask].unsqueeze(-1)

        # find forces
        t = time.time()
        Fa = nf.attractive_force_ahops(D, N, unitD, alpha_hops, k1=1, k2=1)
        durations.t_Fa += time.time()-t

        # Fr = repulsive_force_recip_x(D, N, unitD, k1=2, k2=2)        
        # Fr = nf.repulsive_force_exp(D, N, unitD, k1=10, k2=0.9)
        t = time.time()
        Fr = nf.repulsive_force_hops_exp(D, N, unitD, torch.tensor(hops).to(device), k1=10, k2=0.9)
        durations.t_Fr += time.time()-t
        
        t = time.time()
        F = (Fa+Fr)*random_select
        durations.t_F += time.time()-t
        
        # find acceleration on each point a = F/m
        a = torch.where(mass[:, None] != 0, F / mass[:, None], torch.zeros_like(F))
        # a = np.divide(F, mass[:, None], out=np.zeros_like(Fa), where=mass[:, None]!=0)

        # finally apply relocations
        Z_0 = Z.detach().clone() # save current points
        Z += rate*(a + relocation_noise)

        # get stats
        s = make_embedding_stats(N.cpu().numpy(), hops, maxhops)
        
        # relocations
        relocs = torch.norm(Z - Z_0, dim=1)
        s['relocs-sum'] = torch.sum(relocs)
        s['relocs-mean'] = torch.mean(relocs)
        s['relocs-std'] = torch.std(relocs)

        # relocation noise
        s['reloc-noise'] = 0
        if(args.add_noise):
            s['reloc-noise'] = torch.sum(torch.norm(relocation_noise, dim=1))

        # weighted relocations
        relocs = torch.norm(Z - Z_0, dim=1)*mass
        s['wrelocs-sum'] = torch.sum(relocs)
        s['wrelocs-mean'] = torch.mean(relocs)
        s['wrelocs-std'] = torch.std(relocs)

        # attractive forces
        force = torch.norm(Fa, dim=1)
        s['fa-sum'] = torch.sum(force)
        s['fa-mean'] = torch.mean(force)
        s['fa-std'] = torch.std(force)

        # repulsive forces
        force = torch.norm(Fr, dim=1)
        s['fr-sum'] = torch.sum(force)
        s['fr-mean'] = torch.mean(force)
        s['fr-std'] = torch.std(force)

        # sum of all forces, expected to converge to 0
        s['f-sum-all'] = torch.norm(torch.sum(Fa+Fr, dim=0)) 

        # sum of norm/magnitude of all forces, expected to converge
        s['f-all-magnitude'] = s['fa-sum'] + s['fr-sum']

        s['std-noise'] = std_noise
        if(random_drop.enabled):
            s['selected-ratio'] = torch.sum(random_select)/torch.numel(random_select)
        else:
            s['selected-ratio'] = 1.0
        

        # convert all torch.Tensor elements to regular python numbers
        for k,v in s.items():
            if(type(v) is torch.Tensor):
                s[k] = v.item()
        
        if(len(statsdf) == 0):
            statsdf = pd.DataFrame(columns=list(s.keys()))
            
        s_df = pd.DataFrame(s, index=[0])
        statsdf = pd.concat([statsdf, s_df], ignore_index=True)
        
        # make reports
        aforce = torch.norm(Fa, dim=1)
        logstr+= f"attr:{torch.sum(aforce):<9.3f}({torch.mean(aforce):.3f})  "
        
        rforce = torch.norm(Fr, dim=1)
        logstr+= f"repl:{torch.sum(rforce):<9.3f}({torch.mean(rforce):.3f})  "
        
        force_sum = torch.norm(torch.sum(Fa+Fr, dim=0))
        logstr+= f"sum:{force_sum:<9.3f}  "
        
        logstr+= f"{torch.sum(aforce) + torch.sum(rforce):9.3f} "
        
        relocs = torch.norm(Z - Z_0, dim=1)
        logstr+= f"relocs ({relocs.mean():<6.3f},{relocs.std():<6.3f}) "
        logstr+= f"err-relocs:{s['reloc-noise']:9.3f}\n"

        _n = durations.iteration
        print(f"{_n:6d}| {durations.t_pairwisediff:8.3f} {durations.t_pairwisenorm:8.3f} {durations.t_Fa:8.3f} {durations.t_Fr:8.3f} ")
        print(f"{durations.elapsed():6.1f}| {durations.t_pairwisediff/_n:8.3f} {durations.t_pairwisenorm/_n:8.3f} {durations.t_Fa/_n:8.3f} {durations.t_Fr/_n:8.3f} ")
        
        print(logstr)
        with open(args.logfilepath, "a") as f: # write to the logfile
            f.write(logstr)

        # save the data
        if((_iter)%10==9):
            # save embeddings
            with open(ofilepath, "wb") as f: # save the embedding
                np.save(Z.cpu().numpy(), f)
            with open(histfilepath, "ab") as f: # append embeddings
                pickle.dump(Z.cpu().numpy(), f)
            
            # save stats
            # Save DataFrame to a CSV file
            temp_filename = f"{args.outputdir}/{args.statsfilename}.tmp"
            statsdf.to_csv(temp_filename, index=False)
            # Rename the temporary file to the final filename
            final_filename = f"{args.outputdir}/{args.statsfilename}"
            os.rename(temp_filename, final_filename)

    # The last saving round
    # save embeddings
    with open(ofilepath, "wb") as f: # save the embedding
        np.save(Z.cpu().numpy(), f)
    with open(ofilepath, "ab") as f: # append the last embedding
        pickle.dump(Z.cpu().numpy(), f)
    
    # save stats
    temp_filename = f"{args.outputdir}/{args.statsfilename}.tmp"
    statsdf.to_csv(temp_filename, index=False)
    # Rename the temporary file to the final filename
    final_filename = f"{args.outputdir}/{args.statsfilename}"
    os.rename(temp_filename, final_filename)

    return Z

Z = embed_forcedirected(Z, degrees=degrees, hops=hops)

exit()
# %%
# Open myfile.bin
with open(ofilepath, "rb") as file:
    while True:
        try:
            arr = pickle.load(file)
            print(arr.mean())
        except EOFError:
            break
