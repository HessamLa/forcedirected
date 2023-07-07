import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx

def drawgraph_2d(G, pos, nodes, degrees, figsize=(20,12), sizeratio=None, savepath=''):
    plt.scatter(x_pca.loc[:, 0], x_pca.loc[:, 1], s=np.log(degrees+1))
    
    # drawGraph_forcedirected(G, x_pca.to_numpy(), Fa=Fa, Fr=Fr, with_labels=False)
    # drawGraph_forcedirected(G, x_pca.to_numpy(), with_labels=False)
    
    pos = {nodes[i]:[x_pca.loc[i, 0], x_pca.loc[i, 1]] for i in range(len(nodes))}
    # n_color = np.asarray([degrees[n] for n in nodes])
    n_color = np.asarray(degrees)
    n_size = np.power(n_color,0.8)*sizeratio
    
    fig = plt.figure(figsize=figsize)
    nx.draw_networkx(G, pos=pos, with_labels=False, 
                    node_size=n_size, width=0.05*sizeratio)


    sc = nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes, node_color=n_color, cmap='viridis',
                    # with_labels=False, 
                    node_size=n_size)
    # use a log-norm, do not see how to pass this through nx API
    # just set it after-the-fact
    import matplotlib.colors as mcolors
    sc.set_norm(mcolors.LogNorm())
    fig.colorbar(sc)
    if(len(savepath)>0):
        plt.save
