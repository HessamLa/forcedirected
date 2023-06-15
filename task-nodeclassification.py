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

from utilities import load_graph_networkx, process_graph_networkx

Gx, data = load_graph_networkx(dataset='cora')
G, A, degrees, hops = process_graph_networkx(Gx)
n = A.shape[0]
# find max hops
hops[hops>n+1]=n+1
maxhops = max(hops[hops<n+1])
print("max hops:", maxhops)

# %%
print(type(data), data.y[0])
print(list(set(data.y.tolist())))
print(type(Gx[0]))



# %%
# ?????????? IS G AND Gx THE SAME?
cc = nk.components.ConnectedComponents(G)
cc.run()

components = cc.getComponents()
print(len(components))
# for i, c in enumerate(components):
#     print(f"{i+1:2d}", len(c))
for nodeid in [0,1,2,3,4]:
    print(f"nodeid:{nodeid} nk degree:{G.degree(nodeid)}, nx degree:{Gx.degree(nodeid)}")
    print([n for n in Gx.neighbors(nodeid)])
    print([n for n in G.iterNeighbors(nodeid)])
print(components[0][:10])
print(components[1][:10])
print(components[2][:10])
print(components[3][:10])
print(components[4][:10])
# %%
############################################################################
# LOAD EMBEDDINGS
############################################################################
Z = np.load("embeddings2_.npy")
n,d = Z.shape

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from itertools import product
import random
# DATASET ##################################
class NodeClassificationDataset (Dataset):
    def __init__(self, G, Z, y, ratio=2.0):
        if(type(Z) is not torch.Tensor):
            Z = torch.tensor(Z)
        if(type(y) is not torch.Tensor):
            y = torch.tensor(y)
        
        self._Z = Z.to(torch.float32)
        self._y = y

        assert (len(self._y) == len(self._Z))

    def classes(self):
        return torch.unique(self._y).tolist()

    def __len__(self):
        return len(self._y)
    
    def __getitem__(self, idx):
        return (self._Z[idx], self._y[idx])


# Create dataset
dataset = NodeClassificationDataset(Gx, Z, data.y, ratio=3.0)

# Split the dataset
train_ratio = 0.60
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

print(dataset.classes())
print(train_dataset, train_data_loader)

# %%
# MODEL #################################
class NodeClassificationModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super(NodeClassificationModel, self).__init__()
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
    def forward(self, x):
        # Forward pass through hidden layers
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        
        # Forward pass through output layer
        x = self.sigmoid(self.output_layer(x))

        return x

# Example usage
num_classes = len(dataset.classes()) 
input_dim = d
hidden_dims = [30, 30]  # List of hidden layer dimensions
output_dim = len(dataset.classes())

model = NodeClassificationModel(
            input_dim=input_dim, 
            hidden_dims=hidden_dims, 
            output_dim=output_dim)

# %%%%%%%
# TESTSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Step 6: Training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

best_val_f1 = 0.0  # Track the best validation F1 score
best_model_path = "best_model_classification.pt"  # Path to save the best model

# Create empty lists for storing loss, accuracy, F1 score, ROC AUC values
train_losses = []
val_losses = []
val_accuracies = []
val_f1_scores = []
val_roc_auc_scores = []

epochs = 3000
for epoch in range(epochs):
    # Training
    model.train()
    running_loss = 0.0

    for inputs, targets in train_data_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    # Calculate average loss for the epoch
    train_loss = running_loss / len(train_dataset)

    train_losses.append(train_loss)

    # Print progress
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    val_targets = []
    val_predictions = []
    val_probs = []

    with torch.no_grad():
        for inputs, targets in val_data_loader:
            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)

            # Get predicted labels and probabilities
            _, predicted_labels = torch.max(outputs, dim=1)
            val_targets.extend(targets.cpu().numpy())
            val_predictions.extend(predicted_labels.cpu().numpy())
            val_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    # Calculate average loss, accuracy for validation set
    val_loss /= len(val_dataset)
    val_accuracy = accuracy_score(val_targets, val_predictions)

    # Convert targets, predictions, and probabilities to numpy arrays
    val_targets = np.array(val_targets)
    val_predictions = np.array(val_predictions)
    val_probs = np.array(val_probs)

    # Calculate F1 score for validation set
    val_f1 = f1_score(val_targets, val_predictions, average='weighted')

    # Calculate ROC AUC score for validation set
    val_roc_auc = roc_auc_score(val_targets, val_probs, multi_class='ovr')

    # Append val loss, accuracy, F1 score, ROC AUC to the lists
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_f1_scores.append(val_f1)
    val_roc_auc_scores.append(val_roc_auc)

    # Print validation metrics
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}, Validation ROC AUC: {val_roc_auc:.4f}")
    
    # Update best model if current validation F1 score is better
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)

# Testing
# Load the best model
model.load_state_dict(torch.load(best_model_path))

model.eval()
test_loss = 0.0
test_targets = []
test_predictions = []
test_probs = []

with torch.no_grad():
    for inputs, targets in test_data_loader:
        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, targets)

        test_loss += loss.item() * inputs.size(0)

        # Get predicted labels and probabilities
        _, predicted_labels = torch.max(outputs, dim=1)
        test_targets.extend(targets.cpu().numpy())
        test_predictions.extend(predicted_labels.cpu().numpy())
        test_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

# Calculate average loss and accuracy for test set
test_loss /= len(test_dataset)
test_accuracy = accuracy_score(test_targets, test_predictions)

# Convert targets, predictions, and probabilities to numpy arrays
test_targets = np.array(test_targets)
test_predictions = np.array(test_predictions)
test_probs = np.array(test_probs)

# Calculate F1 score for test set
test_f1 = f1_score(test_targets, test_predictions, average='weighted')

# Calculate ROC AUC score for test set
test_roc_auc = roc_auc_score(test_targets, test_probs, multi_class='ovr')

# Print test metrics
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}, Test ROC AUC: {test_roc_auc:.4f}")

# Plotting
# Loss rate plot
plt.figure()
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Classification Training and Validation Loss')
plt.legend()
# plt.savefig("classification-loss-rate.pdf")
plt.savefig("classification-loss-rate.png")
plt.show()

# Accuracy plot
plt.figure()
plt.plot(range(1, epochs + 1), val_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Classification Validation Accuracy')
# plt.savefig("classification-accuracy.pdf")
plt.savefig("classification-accuracy.png")
plt.show()

# F1 score plot
plt.figure()
plt.plot(range(1, epochs + 1), val_f1_scores)
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Classification Validation F1 Score')
# plt.savefig("classification-f1-score.pdf")
plt.savefig("classification-f1-score.png")
plt.show()

# ROC curve plot
fpr = dict()
tpr = dict()
roc_auc = dict()
for class_idx in range(num_classes):
    fpr[class_idx], tpr[class_idx], _ = roc_curve(test_targets == class_idx, test_probs[:, class_idx])
    roc_auc[class_idx] = roc_auc_score(test_targets == class_idx, test_probs[:, class_idx])

plt.figure()
for class_idx in range(num_classes):
    plt.plot(fpr[class_idx], tpr[class_idx], label=f'Class {class_idx} (AUC = {roc_auc[class_idx]:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Classification ROC Curves')
plt.legend()
# plt.savefig("classification-roc-curves.pdf")
plt.savefig("classification-roc-curves.png")
plt.show()

# # Average ROC curve plot
# micro_fpr, micro_tpr, _ = roc_curve(test_targets.ravel(), test_probs.ravel())
# micro_roc_auc = roc_auc_score(test_targets, test_probs, multi_class='ovr')

# plt.figure()
# plt.plot(micro_fpr, micro_tpr, label=f'Micro-average (AUC = {micro_roc_auc:.2f})')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Classification Average ROC Curve')
# plt.legend()
# plt.savefig("classification-roc-average.pdf")
# plt.savefig("classification-roc-average.png")
# plt.show()


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
