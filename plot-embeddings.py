# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx

# %%
dataset = 'cora'
dataset = 'ego-facebook'
csvpath = f'./embeddings/{dataset}/stats.csv'
df = pd.read_csv(csvpath)
print(df.head())

meancols = [col for col in df.columns 
            if col.endswith('mean') and 
            not col.startswith('hopsinf')]
# print(meancols)
stdcols = [col for col in df.columns 
           if col.endswith('std') and 
           not col.startswith('hopsinf')]
# print(meancols)

ax = df[meancols].iloc[-400:].plot(title=dataset)

# Retrieve the line colors used in the plot
line_colors = [line.get_color() for line in ax.get_lines()]

# Define the legend with colored circles matching the line colors
legend_labels = df[meancols].columns
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) 
                   for color in line_colors]

# Place the legend right to the figure, out of the axis
ax.legend(reversed(legend_elements), reversed(legend_labels), loc='center left', bbox_to_anchor=(1, 0.5))
# Enable gridlines on major ticks
ax.grid(True)
ax.set_ylim(bottom=0.0)
# Adjust the layout to accommodate the legend
# plt.subplots_adjust(right=0.75)

# Show the plot
plt.show()

# %%
from sklearn.decomposition import PCA
def draw_2d(G, Z, nodes, degrees, figsize=(20,12), sizeratio=None):
    plt.close("all")
    if(sizeratio is None):
        sizeratio = (figsize[0]//10)**2
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(Z)
    x_pca = pd.DataFrame(x_pca)
    
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

from utilities import load_graph_networkx, process_graph_networkx

Gx, data = load_graph_networkx(datasetname=dataset, rootpath='../datasets')
G, A, degrees, hops = process_graph_networkx(Gx)

embedpath = f'./embeddings/{dataset}/mbd.npy'
G, A, degrees, hops = process_graph_networkx(Gx)
Z = np.load(embedpath)
draw_2d(Gx, Z, list(Gx.nodes), degrees)

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
selecttag = 'rselect'
dataset = 'ego-facebook'

csvpath = f'./embeddings-test/{dataset}-{noisetag}-{selecttag}/stats.csv'

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