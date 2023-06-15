# %%
# %reload_ext autoreload
# %autoreload 2
import networkit as nk
import matplotlib.pyplot as plt
import math
import numpy as np

import networkx as nx
import sklearn as sk
import pandas as pd

import torch
from utils import drawGraph_forcedirected
from sklearn.metrics.pairwise import euclidean_distances
# %%
############################################################################
# Load data
############################################################################
# Get the Cora dataset
datasetname = 'cora'
datasetname = 'ego-facebook'
rootpath = '../datasets'
embeddingpath = f"embeddings/{datasetname}/mbd.npy"

from utilities import load_graph_networkx, process_graph_networkx

Gx, data = load_graph_networkx(datasetname=datasetname, rootpath=rootpath)
G, A, degrees, hops = process_graph_networkx(Gx)
n = A.shape[0]
# find max hops
hops[hops>n+1]=n+1
maxhops = max(hops[hops<n+1])
print("max hops:", maxhops)


# %%
# ?????????? IS G AND Gx THE SAME?
cc = nk.components.ConnectedComponents(G)
cc.run()

components = cc.getComponents()
print(len(components))
# # for i, c in enumerate(components):
# #     print(f"{i+1:2d}", len(c))
# for nodeid in [0,1,2,3,4]:
#     print(f"nodeid:{nodeid} nk degree:{G.degree(nodeid)}, nx degree:{Gx.degree(nodeid)}")
#     print([n for n in Gx.neighbors(nodeid)])
#     print([n for n in G.iterNeighbors(nodeid)])
# print(components[0][:10])
# print(components[1][:10])
# print(components[2][:10])
# print(components[3][:10])
# print(components[4][:10])
# %%
############################################################################
# LOAD EMBEDDINGS
############################################################################
Z = np.load(embeddingpath)
n,d = Z.shape



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from itertools import product
import random
# DATASET ##################################
class LinkPredictionDataset (Dataset):
    def __init__(self, G, Z, ratio=2.0):
        if(type(Z) is not torch.Tensor):
            Z = torch.tensor(Z)
        
        self._Z = Z.to(torch.float32)

        # generate positive edges
        self._edges = [(u, v, 1) for u, v in G.edges()]
        
        # generate negative edges, with ratio 2 negative to 1 positive
        negative_count = int(len(self._edges)*ratio)
        all_negative_edges = list(nx.non_edges(G))
        self._edges0 = random.sample(all_negative_edges, negative_count) # negative edges
        self._edges0 = [(u, v, 0) for u, v in self._edges0]

        self._all_edges = list(set(self._edges+self._edges0))
        random.shuffle(self._all_edges)

    def get_positive_edges (self):
        return self._edges
    
    def __str__(self) -> str:
        return super().__str__(self.__repr__())
    
    def __repr__(self) -> str:
        return super().__repr__(f"positive edges:{len(self.edegs)}, negative edges:{len(self._edges0)}")

    def __len__(self):
        return len(self._all_edges)
    
    def __getitem__(self, idx):
        i,j,target = self._all_edges[idx]

        # t = torch.cat((self.Z[i], self.Z[j]))
        # print(idx, i, j, self.Z[i,:].size(), t.size())
        return torch.cat((self._Z[i], self._Z[j])), target


# Create dataset
dataset = LinkPredictionDataset(Gx, Z, ratio=3.0)

# Split the dataset
train_ratio = 0.6
val_ratio = 0.20
test_ratio = 0.20

# Calculate sizes of each split
train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
batch_size = 1024
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
for xx in train_data_loader:
    print(type(xx), len(xx), xx[0].size())
    print(type(xx[0]), type(xx[1]))
    print(xx[0][0])
    print(xx[1][0])
    break

print(train_dataset, train_data_loader)
# %%
# MODEL #################################
class LinkPredictionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super(LinkPredictionModel, self).__init__()
        self.hidden_layers = nn.ModuleList()

        # Add input layer
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Add hidden layers
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        # Add output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        # Forward pass through hidden layers
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
            x = self.dropout(x)
        
        # Forward pass through output layer
        x = self.sigmoid(self.output_layer(x))

        return x

# Example usage
input_dim = 2*d
hidden_dims = [60, 60]  # List of hidden layer dimensions
output_dim = 1

model = LinkPredictionModel(
            input_dim=input_dim, 
            hidden_dims=hidden_dims, 
            output_dim=output_dim)

# Step 6: Training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()


# %%%%%%%
# TESTSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

# Step 6: Training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Create empty lists for storing loss and AUC values
train_losses = []
val_losses = []
val_auc_values = []

epochs = 1000
for epoch in range(epochs):
    # Training
    model.train()
    running_loss = 0.0

    i_batch = 0
    i_n_batch = 0

    for i, batch in enumerate(train_data_loader):
        inputs, targets = batch
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate the loss
        loss = criterion(outputs.squeeze(), targets.float())
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)

        # i_batch += 1
        # i_n_batch += len(inputs)
        # print(i_batch, i_n_batch)
        
        
    # Calculate average loss for the epoch
    train_loss = running_loss / len(train_dataset)
    
    train_losses.append(train_loss)
    
    # Print progress
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_targets = []
    val_outputs = []

    with torch.no_grad():
        for batch in val_data_loader:
            inputs, targets = batch
            # Forward pass
            outputs = model(inputs).squeeze()
            
            # Calculate the loss
            loss = criterion(outputs, targets.float())
            
            val_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            predicted_labels = torch.round(outputs)
            val_correct += (predicted_labels == targets).sum().item()

            # Accumulate targets and outputs for AUC calculation
            val_targets.extend(targets.cpu().numpy())
            val_outputs.extend(outputs.cpu().numpy())

    
    # Calculate average loss and accuracy for validation set
    val_loss /= len(val_dataset)
    val_accuracy = val_correct / len(val_dataset)

    # Convert targets and outputs to numpy arrays
    val_targets = np.array(val_targets)
    val_outputs = np.array(val_outputs)

    # Calculate AUC and F1 score for validation set
    val_auc = roc_auc_score(val_targets, val_outputs)
    val_f1 = f1_score(val_targets, np.round(val_outputs))

    # Append val loss and AUC to the lists
    val_losses.append(val_loss)
    val_auc_values.append(val_auc)

    # Print validation metrics
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation AUC: {val_auc:.4f}, Validation F1 Score: {val_f1:.4f}")


# Testing
model.eval()
test_loss = 0.0
test_correct = 0
test_targets = []
test_outputs = []

with torch.no_grad():
    for batch in test_data_loader:
        inputs, targets = batch
        
        # Forward pass
        outputs = model(inputs).squeeze()
        
        # Calculate the loss
        loss = criterion(outputs, targets.float())
        
        test_loss += loss.item() * inputs.size(0)
        
        # Calculate accuracy
        predicted_labels = torch.round(outputs)
        test_correct += (predicted_labels == targets).sum().item()
        
        # Accumulate targets and outputs for AUC calculation
        test_targets.extend(targets.cpu().numpy())
        test_outputs.extend(outputs.cpu().numpy())

# Calculate average loss and accuracy for test set
test_loss /= len(test_dataset)
test_accuracy = test_correct / len(test_dataset)

# Convert targets and outputs to numpy arrays
test_targets = np.array(test_targets)
test_outputs = np.array(test_outputs)

# Calculate AUC and F1 score for test set
test_auc = roc_auc_score(test_targets, test_outputs)
test_f1 = f1_score(test_targets, np.round(test_outputs))

# Print test metrics
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}, Test F1 Score: {test_f1:.4f}")

# %%

# Plotting
# Loss rate plot
plt.figure()
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Link Prediction Training and Validation Loss')
plt.legend()
plt.savefig(f"{datasetname}-lp-loss-rate.pdf")
plt.savefig(f"{datasetname}-lp-loss-rate.png")
plt.show()

# AUC curve plot
fpr, tpr, thresholds = roc_curve(test_targets, test_outputs)
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Link Prediction ROC Curve')
plt.savefig(f"{datasetname}-lp-roc-curve.pdf")
plt.savefig(f"{datasetname}-lp-roc-curve.png")
plt.show()

# %%
############################################################################
# Embed
############################################################################
# lim = 100
# run_forcedirected(Z[:lim], mass[:lim], hops[:lim, :lim], alpha=0.3)

if __name__ == "__main__":
    exit()
############################################################
############################################################
# %%
############################################################


# %%

# load the saved points
import nodeforce as nf
# %%
# points = np.load("temp-points_.npy")
points = np.load("embeddings2_.npy")
# Calculate forces
def verify_random_points(points):
    distances = np.linalg.norm(points, axis=1)
    
    # Plotting histogram
    plt.hist(distances, bins=20)
    plt.xlabel('Distance from Origin')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distance from Origin')
    plt.show()
verify_random_points(points)
n,d = points.shape
print(f"A matrix of shape {points.shape} where each row corresponds to embedding of a node in a {d}-dimensional space.")

D = nf.pairwise_difference(torch.Tensor(points))
N = torch.norm(D, dim=-1)

_hops = hops.copy()
print(hops.shape)

print(D.min())
print(torch.mean(N), torch.std(N))
print(_hops.shape)
print(_hops[0,1])

print(N.size(), hops.shape)
ht = _hops[_hops<=maxhops]
Nt = N.numpy()
print(Nt.shape, ht.shape)
for l in range(1, int(ht.max()+1)):
    mask = _hops==l
    print(f"{l:3d} {Nt[mask].mean():10.3f} {Nt[mask].std():10.3f} {len(Nt[mask])/2:8.0f}")
# disconnected components
mask = _hops>20
print(f"inf {Nt[mask].mean():10.3f} {Nt[mask].std():10.3f} {len(Nt[mask])/2:8.0f}")
    
mask = torch.triu(torch.ones_like(N), diagonal=1)
print(torch.min(N[mask.bool()]))

# %%
Z = points.copy()
plt.close("all")
# %%
v = 0
nv = cc.componentOfNode(v)
print(nv)
print(cc.getComponents()[0])
idx = A==1
print(A[idx])


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
    n_size = n_color*sizeratio
    
    fig = plt.figure(figsize=figsize)
    nx.draw_networkx(G, pos=pos, with_labels=False, 
                    node_size=sizeratio, width=0.05*sizeratio)


    sc = nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes, node_color=n_color, cmap='viridis',
                    # with_labels=False, 
                    node_size=n_color*sizeratio)
    # use a log-norm, do not see how to pass this through nx API
    # just set it after-the-fact
    import matplotlib.colors as mcolors
    sc.set_norm(mcolors.LogNorm())
    fig.colorbar(sc)

draw_2d(Gx, Z, list(Gx.nodes), degrees)
# %%
#Draw
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(Z)
x_pca = pd.DataFrame(x_pca)
x_pca.head()  

# print(list(x_pca.columns))
# %% draw 2d

pca = PCA(n_components=2)
x_pca = pca.fit_transform(Z)
x_pca = pd.DataFrame(x_pca)
draw_2d(Gx, x_pca.to_numpy, list(Gx.nodes), degrees)
# plt.scatter(x_pca.loc[:, 0], x_pca.loc[:, 1], s=np.log(degrees+1))
# # plt.scatter(x_pca.loc[:, 0], x_pca.loc[:, 1], s=degrees/10)
# fig=plt.figure()

# # drawGraph_forcedirected(G, x_pca.to_numpy(), Fa=Fa, Fr=Fr, with_labels=False)
# # drawGraph_forcedirected(G, x_pca.to_numpy(), with_labels=False)
# nx.draw_networkx(Gx, x_pca.to_numpy(), with_labels=False, 
#                  node_size=.01, width=1.)


# %%
# draw the largest component
cc = nk.components.ConnectedComponents(G)
cc.run()
components = cc.getComponents()
if(len(components)>1):
    nodes = components[0]
    for i in range(len(components)):
        if(len(components[i])>len(nodes)):
            nodes = components[i]
    Z0 = Z[nodes,...]
    Gx0 = Gx.subgraph(nodes)
    # pos = {nodes[i]:[x_pca.loc[i, 0], x_pca.loc[i, 1]] for i in range(len(nodes))}
    
    draw_2d(Gx0, Z0, list(Gx0.nodes), degrees[nodes])
    
    
# %%
x2 = x_pca.copy()
x2.loc[:,1] = x2.loc[:,1]*100 # expand along y axis
drawGraph_forcedirected(G, x2.to_numpy(), with_labels=False)

# %% draw 3d
pca = PCA(n_components=3)
x_pca = pca.fit_transform(Z)
x_pca = pd.DataFrame(x_pca)

# %%
plotdata = []
y = data.y.numpy().astype(int)
for i in np.unique(y):
    idx = y==i
    d = (x_pca.loc[idx, 0], x_pca.loc[idx, 1], x_pca.loc[idx, 2], y[idx])
    plotdata.append(d)

# %%
# get the 3d data
class_label = data.y.numpy().astype(int)
# draw the 3d data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x1,x2,x3 = (x_pca.loc[:, 0], x_pca.loc[:, 1], x_pca.loc[:, 2])
y = data.y.numpy().astype(int)
labels = [f'Class {i}' for i in np.unique(y)]
scatter = ax.scatter(x1, x2, x3, c=y, cmap='tab10', s=1)
ax.legend(handles=scatter.legend_elements()[0], labels=labels,
          loc='upper left', numpoints=1, ncol=4, fontsize=8, bbox_to_anchor=(0, 0))
plt.show()
# colors = []
# for i in np.unique(y):
#     idx = y==i
#     scatter = ax.scatter(x1[idx], x2[idx], x3[idx], c=y[idx], cmap='tab10', label=f'Class {i}', s=1)
# ax.legend()

# ax.grid(True)

# %%
# Create the figure and 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot for each class
scatter_plots = []
for i, (x1, x2, x3, class_label) in enumerate(plotdata):
    scatter = ax.scatter(x1, x2, x3, c=class_label, cmap='tab10', label=f'Class {i}', s=1, )
    scatter_plots.append(scatter)
# # Set the viewing angles

# ax.view_init(elev=30, azim=45)

# # Set the labels and legend
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

print(np.unique(y))
# %%

# Set the viewing angles
ax.view_init(elev=30, azim=45)

# Set the labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Create a checkbox widget to select which classes to show
class_checkboxes = []
for i, scatter in enumerate(scatter_plots):
    class_checkbox = plt.axes([0.02, 0.9 - (i * 0.05), 0.1, 0.03])
    class_check = plt.Checkbutton(class_checkbox, f'Class {i}', visible=True)
    class_check.on_clicked(lambda event, scatter=scatter: scatter.set_visible(not scatter.get_visible()))
    class_checkboxes.append(class_check)

plt.show()
# %%
plt.close('all')
fig = plt.figure()
idx = data.y.numpy()==1
ax = fig.add_subplot(projection='3d')
ax.scatter(x_pca.loc[idx, 0], x_pca.loc[idx, 1], x_pca.loc[idx, 2],
           c=data.y[idx],
           s=np.log(degrees[idx]+1)/10)
# drawGraph_forcedirected(G, x_pca.to_numpy(), Fa, Fr, distance_scale,with_labels=False, draw_attractions=False, draw_repulsions=False)

# dpi = 300  # Specify the DPI value (e.g., 300 for high resolution)
# output_file = 'high_res_plot.png'  # Specify the output filename and extension
# fig.savefig(output_file, dpi=dpi)

ax.view_init(elev=30, azim=45)  # First viewing angle
plt.savefig('3d_plot_1.png')  # Save the plot

ax.view_init(elev=15, azim=-30)  # Second viewing angle
plt.savefig('3d_plot_2.png')  # Save the plot

ax.view_init(elev=60, azim=120)  # Third viewing angle
plt.savefig('3d_plot_3.png')  # Save the plot

# %%
