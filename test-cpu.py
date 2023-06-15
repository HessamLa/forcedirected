import torch
from torch import nn
from torch import Tensor
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import WebKB

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if(torch.cuda.is_available()):
    print("cuda is available")
print(device)
