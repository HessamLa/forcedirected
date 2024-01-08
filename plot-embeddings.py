# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.decomposition import PCA

import networkx as nx
from forcedirected.utilities.graphtools import process_graph_networkx

from forcedirected.utilities import RecursiveNamespace as rn

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

def reducedim (Z, target_dim, method='PCA', **kwargs):
    if(method.upper() == 'PCA'):
        pca = PCA(n_components=target_dim, **kwargs)
        X = pca.fit_transform(Z)
        return X
    else:
        print(method, 'IS UNKNOWN')
        raise

# %%

dataset = 'pubmed'
dataset = 'ego-facebook'
args = rn({'edgelist': f'./datasets/{dataset}/{dataset}_edgelist.txt',
           'labels':f'./datasets/{dataset}/{dataset}_y.txt'})
print(args)
Gx = load_graph(args)
# get node degrees
degrees = np.asarray([Gx.degree(n) for n in Gx.nodes])
# if args.labels path exists, load labels
labels=None
if('labels' in args and os.path.exists(args.labels)):
    labels = np.loadtxt(args.labels, dtype=str)
# %%
print(args)    
def drawgraph_2d(G, Z, nodes, degrees, figsize=None, figheight=10,
        sizeratio=None, title=None, ax=None, add_colorbar=True,
        cmap=None, node_color=None,
        ):
    plt.close("all")

    X = reducedim(Z, 2)
    # find width and height of the bounding box
    if(figsize is None):
        w = X[:,0].max() - X[:,0].min()
        h = X[:,1].max() - X[:,1].min()
        figsize = (figheight*w/h, figheight)  
    if(sizeratio is None):
        sizeratio = figsize[0]**2.5/100

    
    nodes = list(G.nodes)
    pos = {nodes[i]:[X[i, 0], X[i, 1]] for i in range(len(nodes))}
    
    # n_color = np.asarray([degrees[n] for n in nodes])
    viridis = mpl.colormaps['viridis'].resampled(128)
    
    n_color = np.asarray(degrees)
    if(cmap is None):
        cmap = mpl.colormaps['viridis'].resampled(128)
        # cmap = viridis(0.5+degrees/(max(degrees)*2))
    n_size = np.power(n_color,0.7)*sizeratio

    if(ax is None):
        print("Create new matplotlib axis with size", figsize)
        # Modify here
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.axis('off')
    else:
        plt.sca(ax)
        fig = ax.figure
        print("Use the figure with size", fig.figsize)
    
    fig.tight_layout()

    # remove grid from axis
    ax.grid(False)    

    # Set vmin and vmax to use only upper half of colormap
    vmin = .0
    vmax = .999
    
    if(node_color is None):
        node_color=np.sqrt(np.asarray(degrees)/max(degrees))
        node_color=0.05+node_color*0.95
    # vmin = 2
    # vmax = max(degrees)/1.1
    # vmax = 20
    # degree_ranges=np.asarray(degrees)
    # offset=1
    # degree_ranges=(1-offset+degree_ranges*offset) # offset
    
    # plt.scatter(X[:, 0], X[:, 1], s=np.log(degrees+1))
    nx.draw_networkx(G, pos=pos, with_labels=False, 
                    node_size=n_size, width=0.05*sizeratio,
                    node_color=node_color, 
                    cmap=cmap, 
                    vmin=vmin, vmax=vmax)
    # make tight axis
    # ax.set_aspect('equal')
    mpl.rcParams['lines.linewidth'] = 0.001*sizeratio

    # add title to ax
    if(title is not None):
        ax.set_title(title)
    
    return ax.figure

# %%
method='forcedirected_v0106_128d'
method='forcedirected_v0107_128d'
method='forcedirected_v0108_128d'
method='forcedirected_v0109_128d'
method='forcedirected_v0110_128d'
method='ge-deepwalk_128d'
method='ge-node2vec_128d'
method='forcedirected_v0120_128d'
method='forcedirected_v0104_128d'
method='forcedirected_v0121_128d'
method='forcedirected_v0201_128d'
args.embedding=f'./embeddings-tmp/{method}/{dataset}/embed-df.pkl.tmp'
args.embedding=f'./embeddings-tmp/{method}/{dataset}/embed-df.pkl'

# Load embeddings
Z = pd.read_pickle(args.embedding)
Z = Z.to_numpy()
print(Z.shape)

# fig, axes = plt.subplots(1,1, figsize=(10,6))

# Get the number of labels
# num_labels = len(np.unique(labels[:,1]))
# # Generate color map
# cmap = plt.cm.get_cmap('Set1', num_labels) 
node_color=None
if(labels is not None):
    node_color = labels[:,1].astype(int)
    # divide node_color array by the max
    node_color = node_color / max(node_color)
title=''
# title=f'{method} embedding {dataset}'

fig = drawgraph_2d(Gx, Z, list(Gx.nodes), degrees, 
                   title=title,
                   figheight=10,
                   node_color=node_color
                   )
plt.savefig(f'./images/plot-embeddings-{method}-{dataset}.pdf')
plt.savefig(f'./images/plot-embeddings-{method}-{dataset}.svg')
plt.savefig(f'./images/plot-embeddings-{method}-{dataset}.png')

# fig.show()

# %%
G, A, degrees, hops = process_graph_networkx(Gx)
print('processed graph')
# %%
dataset_args=rn()
dataset = 'ego-facebook'
method='forcedirected_v0005_128d'
method='ge-node2vec_128d'
method_name={'forcedirected_v0005_128d':'Force-Directed', 'ge-node2vec_128d':'Node2Vec', 'ge-deepwalk_128d':'DeepWalk'}
for method in ['forcedirected_v0005_128d', 'ge-node2vec_128d', 'ge-deepwalk_128d']:
    dataset_args[method] = rn({
            'edgelist': f'./datasets/{dataset}/{dataset}_edgelist.txt',
            'embedding':f'./embeddings/{method}/{dataset}/embed-df.pkl',
            'methodname': method_name[method],
            })
print(dataset_args)

fig, axes = plt.subplots(1,3, figsize=(21,6))
for ax, args in zip(axes, dataset_args.values()):
    print(ax)
    print(args)
    Gx = load_graph(args)
    G, A, degrees, hops = process_graph_networkx(Gx)
    print('processed graph')

    Z = pd.read_pickle(args.embedding)
    Z = Z.to_numpy()
    print(Z.shape)

    _ = drawgraph_2d(Gx, Z, list(Gx.nodes), degrees, title=f'{args.methodname} embedding {dataset}', ax=ax)
plt.subplots_adjust(wspace=0, hspace=0)
plt.subplots_adjust(top=0.85)
plt.tight_layout()
plt.savefig(f'./images/2d-multi.pdf')
fig.show()



# %%

embedpath = f'./embeddings/{dataset}/embed.npy'
noise,droprate = '','none'
# noise = 'noise'
droprate = 'steady-rate'
embedpath = f'./embeddings-drop-test/{dataset}-{noise}-{droprate}/embed.npy'
# G, A, degrees, hops = process_graph_networkx(Gx)

Z = np.load(embedpath)
print(Z.shape)
drawgraph_2d(Gx, Z, list(Gx.nodes), degrees, title=f'NodeForce {dataset}-{noise}-{droprate}')
plt.savefig(f'./images/NodeForce-{dataset}-{noise}-{droprate}.png')


# %%
Gx, data = load_graph_networkx(datasetname='corafull', rootpath='./datasets')
nx.write_edgelist(Gx, path='./datasets/corafull/corafull.edgelist')
nx.write_adjlist(Gx, path='./datasets/corafull/corafull.adjlist')
# %%
method='node2vec'
method='nodeforce'
method='deepwalk'
dataset='ego-facebook'
dataset='pubmed'
for dataset in ['cora', 'ego-facebook']:
    try:
        Gx, data = load_graph_networkx(datasetname=dataset, rootpath='./datasets')
        print('loaded graph')
    except:
        print('faile loading', dataset)
        G, A, degrees, hops = process_graph_networkx(Gx)
        print('processed graph')
        continue
    G, A, degrees, hops = process_graph_networkx(Gx)
    for method in ['node2vec', 'nodeforce', 'deepwalk']:
        try:
            embedpath = f'./embeddings/{method}/{dataset}/embed.npy'
            Z = np.load(embedpath)
            # print(Z.shape)
        except:
            print(f'faile loading embedding for {dataset} with {method} method')
            continue
        drawgraph_2d(Gx, Z, list(Gx.nodes), degrees, title=f'{method} {dataset}')
        plt.tight_layout()
        plt.savefig(f'./images/2d-{dataset}-{method}.png')

# %%
# # get embedding stats
# import torch
# from nodeforce import pairwise_difference
# from utilities import make_embedding_stats
# N = pairwise_difference(torch.tensor(Z))
# n = Z.shape[0]
# s = make_embedding_stats(N, hops, max(hops[hops<n+1]))
# print(s)
# %% draw 3d
from matplotlib.animation import FuncAnimation
pca = PCA(n_components=3)
x_pca = pca.fit_transform(Z)
x_pca = pd.DataFrame(x_pca)
# get the 3d data
# draw the 3d data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x1,x2,x3 = (x_pca.loc[:, 0], x_pca.loc[:, 1], x_pca.loc[:, 2])
# y = data.y.numpy().astype(int)
y = np.array([1]*len(x1))

# node color
n_color = y
n_color = np.asarray(degrees)
# node size
n_size = np.power(np.asarray(degrees),0.5)

labels = [f'Class {i}' for i in np.unique(y)]
scatter = ax.scatter(x1, x2, x3, c=n_color, cmap='tab10', s=n_size)
ax.legend(handles=scatter.legend_elements()[0], labels=labels,
          loc='upper left', numpoints=1, ncol=4, fontsize=8, bbox_to_anchor=(0, 0))

# Set the initial view angle for the 3D plot
ax.view_init(elev=30, azim=0)

# Function to update the view angle for each frame of the animation
def update(frame):
    ax.view_init(elev=30, azim=frame * 4)  # Adjust the increment value for rotation speed

# Create the animation
animation = FuncAnimation(fig, update, frames=90, interval=50)  # Adjust the number of frames and interval as desired

# Save the animation as a GIF
animation.save('rotation.gif', writer='pillow')

# Show the plot
# plt.show()
plt.show()
# %%
dataset, noisetag, selecttag = ('cora', '', '')

noisetag = 'noise'
# selecttag = 'rselect'
dataset = 'ego-facebook'

csvpath = f'./embeddings-carb/{dataset}-{noisetag}-{selecttag}/stats.csv'

noisetag = 'noise'
droptag = 'steady-rate'
droptag = 'none'
csvpath = f'./embeddings-drop-test/{dataset}-{noisetag}-{droptag}/stats.csv'

df = pd.read_csv(csvpath)
# print(df.head())


hopsmean = [col for col in df.columns 
            if col.endswith('mean') and 
            col.startswith('hops') and
            not col.startswith('hopsinf')]
# print(meancols)
hopsstd = [col for col in df.columns 
           if col.endswith('std') and 
           col.startswith('hops') and
           not col.startswith('hopsinf')]

fromidx = 1
toidx = -1
# toidx = 125
df[['relocs-mean', 'wrelocs-mean']][fromidx:toidx].plot(title=f"mean relocs {csvpath}")
df[['relocs-std', 'wrelocs-std']][fromidx:toidx].plot(title=f"std relocs {csvpath}")

df[hopsmean][fromidx:toidx].plot(title=f"mean hops {csvpath}")
df[hopsstd][fromidx:toidx].plot(title=f"std hops {csvpath}")
# %%
