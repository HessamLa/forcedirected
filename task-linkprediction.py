# %%
# %reload_ext autoreload
# %autoreload 2
import argparse
from pprint import pprint
from typing import Any
from utilities import ReportLog
DEFAULT_DATASET='cora'
DATASET_CHOICES=['cora', 'citeseer', 'pubmed', 'ego-facebook', 'corafull']
DEFAULT_METHOD='nodeforce'
METHOD_CHOICES=['nodeforce', 'n2v']
# DEFAULT_DATASET='ego-facebook'
parser = argparse.ArgumentParser(description='Process command line arguments.')
parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET, choices=DATASET_CHOICES, 
                    help='name of the dataset (default: cora)')
parser.add_argument('--method', type=str, default=DEFAULT_METHOD, choices=METHOD_CHOICES,
                    help=f"embedding method (default: nodeforce)")
parser.add_argument('--embeddingpath', type=str, default=None, 
                    help='path to the embedding file (default: ./embeddings/{dataset}/{method}/embed.npy)')
parser.add_argument('--outdir', type=str, default='./linkprediction', 
                    help='path to the output directory (default: ./linkprediction)')
parser.add_argument('--logfilepath', type=str, default=None, 
                    help='path to the log file (default: {outdir}/lp-{dataset}-{method}.txt)')
parser.add_argument('--epochs', type=int, default=1000, 
                    help='number of training epochs (default: 1000)')
parser.add_argument('--description', type=str, default="", nargs='+', 
                    help='description, used for experimentation logging')
args, unknown = parser.parse_known_args()
# use default parameter values if required
if(args.embeddingpath is None):
    args.embeddingpath = f"./embeddings/{args.dataset}/{args.method}/embed.npy"
if(args.logfilepath is None):
    args.logfilepath = f"{args.outdir}/lp-{args.dataset}-{args.method}-log.txt"

# Combine the description words into a single string
args.description = ' '.join(args.description)

log = ReportLog(args.logfilepath)
if(len(unknown)>0):
    log.print("====================================")
    log.print("THERE ARE UNKNOWN ARGUMENTS PASSED:")
    log.pprint(unknown)
    log.print("====================================")
   
log.print("\nArguments:")
for key, value in vars(args).items():
    log.print(f"{key:20}: {value}")
log.print("")

# %%


# %%

import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

import networkit as nk
import networkx as nx

import torch
from utils import drawGraph_forcedirected
# %%
############################################################################
# Load data
############################################################################
# Get the Cora dataset
# datasetname = 'corafull'
# datasetname = 'cora'
# datasetname = 'ego-facebook'
# rootpath = './datasets'
# emethod='n2v'
# embeddingpath = f"./embeddings/{datasetname}/{emethod}/embed.npy"

from utilities import load_graph_networkx, process_graph_networkx

Gx, data = load_graph_networkx(datasetname=args.dataset, rootpath='./datasets')
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
log.print("load embedding from", args.embeddingpath)
Z = np.load(args.embeddingpath)
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

epochs = args.epochs
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
    log.print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}")
    # log(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}")

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
    log.print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    log.print(f"Validation AUC: {val_auc:.4f}, Validation F1 Score: {val_f1:.4f}")
    

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
log.print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
log.print(f"Test AUC: {test_auc:.4f}, Test F1 Score: {test_f1:.4f}")

# %%

# Plotting
# Loss rate plot
plt.figure()
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Link Prediction Training and Validation Loss \n(dataset:{args.dataset} method:{args.method})')
plt.legend()
plt.savefig(f"./images/{args.dataset}-{args.method}-lp-loss-rate.pdf")
plt.savefig(f"./images/{args.dataset}-{args.method}-lp-loss-rate.png")
plt.show()

# AUC curve plot
fpr, tpr, thresholds = roc_curve(test_targets, test_outputs)
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Link Prediction ROC Curve \n(dataset:{args.dataset} method:{args.method})')
plt.savefig(f"./images/{args.dataset}-{args.method}-lp-roc-curve.pdf")
plt.savefig(f"./images/{args.dataset}-{args.method}-lp-roc-curve.png")
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