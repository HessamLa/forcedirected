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
from nodeforce import *
# %%
"""
Returns a matrix such that M[i,j] = P[j]-P[i].
Summing row i corresponds with force from i to others
Summing col j corresponds with forces from other to j
"""
def pairwise_difference(P: torch.tensor) -> torch.tensor:
    n, d = P.size()
    # Expand dimensions of P to create row-wise repetitions
    P_row = P.unsqueeze(1)    
    # Expand dimensions of P to create column-wise repetitions
    P_col = P.unsqueeze(0)
    # Compute the matrix M
    # print(P_row.size(), P_col.size())
    D = P_col - P_row
    return D

# %%
"""
hops is numpy.narray
"""
def get_alpha_to_hops (hops, alpha: float):
    hops=hops.astype(float)
    # alpha^(h-1)
    alpha_to_hops = np.power(alpha, hops-1, out=np.zeros_like(hops), where=hops!=0)
    return alpha_to_hops
def test_get_alpha_to_hops ():
    print("Test get_alpha_to_hops(.)")
    hops = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,2,0]])    
    a = get_alpha_to_hops(hops, alpha=0.3)
    print(a)
test_get_alpha_to_hops()
# %%
"""
Given an array of hops, it calculates the coefficient of each corresponding position
and returns a corresponding array.

For position if hops[i] is h, result[i] is as follows:
result[i] = 1/|N_h| * alpha^(h-1)

|N_h| is the total number of elements equal to h in hops array.
Essentially it means the total number of neighbors at minimum h-hop distance.
"""
def get_alpha_hops (hops, alpha: float):
    hops=hops.astype(float)
    
    # alpha^(h-1)
    alpha_to_hops = get_alpha_to_hops(hops, alpha)
    
    unique_values, counts = np.unique(hops, return_counts=True) # number of nodes at i-hop distance
    # |N_h| per entry. For any (i,j) with h-hops distance, abs_Nh[i,j]=s|N_h|
    abs_Nh = counts[np.searchsorted(unique_values, hops)] 
    
    result = alpha_to_hops/abs_Nh
    return result
def test_get_alpha_hops ():
    hops = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,2,0]])
    alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=hops, alpha=0.3)
    print(alpha_hops)
test_get_alpha_hops()
# print(hops)
# print(alpha_hops)
# %%
"""
D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
N (n,n) is norm of each pairwise diff element, i.e x.
unitD (n,n,d) is the unit direction, D/N
k1 is amplitude factor, scalar: k1*f(x) 
k2 is decaying factor factor, scalar: f(x/k2)
"""
def repulsive_force_exp(D, N, unitD, k1=1, k2=1, return_sum=True):
    n = D.shape[0] # total number of nodes
    
    # force amplitudes is the main part the algorithm
    # get the negative directions, for force
    F = -unitD
    # apply amplitudes
    F = F * k1/n * torch.exp(-N/k2).unsqueeze(-1)
    
    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i

        # print("After sum")
        # print("F[2]\n", F[2])

    return F
def test_repulsive_force_exp():
    hops = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,2,0]])
    alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=hops, alpha=0.3)

    a = [[1,1,1], [0,0,0], [2,2,2], [-1,-1,-1]]
    a = torch.tensor(a).float()
    D = pairwise_difference(a)
    N = torch.norm(D, dim=-1)

    # Element-wise division with mask
    unitD = torch.zeros_like(D)
    mask = N!=0 # Create a mask for non-zero division
    unitD[mask] = D[mask] / N[mask].unsqueeze(-1)

    v = D*N.unsqueeze(-1)
    v = unitD*N.unsqueeze(-1)
    print(D.size(), N.size(), unitD.size())
    t = repulsive_force_exp(D, N, unitD)
    print(t)
test_repulsive_force_exp()
# %%
"""
Calculates repulsive force by inverse of the distance
f(x) = k1/(x^k2)

D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
N (n,n) is norm of each pairwise diff element, i.e x.
unitD (n,n,d) is the unit direction, D/N
k1 is amplitude factor, scalar.
k2 is power factor, scalar.
"""
def repulsive_force_recip_x(D, N, unitD, k1=1, k2=1, return_sum=True):
    n = D.shape[0] # total number of nodes
    
    # force amplitudes is the main part the algorithm
    # get the negative directions, for force
    F = -unitD
    
    # apply amplitudes
    x_to_k2 = torch.pow(N, k2) # raise x to power of k2
    F = F * k1/n * torch.where(x_to_k2!= 0, k1 / x_to_k2, torch.zeros_like(x_to_k2)).unsqueeze(-1)
    
    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i

        # print("After sum")
        # print("F[2]\n", F[2])

    return F
def test_repulsive_force_recip_x():
    hops = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,2,0]])
    alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=hops, alpha=0.3)

    a = [[1,1,1], [0,0,0], [2,2,2], [-1,-1,-1]]
    a = torch.tensor(a).float()
    D = pairwise_difference(a)
    N = torch.norm(D, dim=-1)

    # Element-wise division with mask
    unitD = torch.zeros_like(D)
    mask = N!=0 # Create a mask for non-zero division
    unitD[mask] = D[mask] / N[mask].unsqueeze(-1)
    t = repulsive_force_recip_x(D, N, unitD)
    print(t)
test_repulsive_force_recip_x()
# %%
"""
Calculates repulsive force by inverse of the distance
f(x) = k1*alpha^(h-1)/(x^k2)

D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
N (n,n) is norm of each pairwise diff element, i.e x.
unitD (n,n,d) is the unit direction, D/N
alpha, scalar is the coefficient for each hop-distant neighbor
k1 is amplitude factor, scalar.
k2 is power factor, scalar.
"""
def repulsive_force_alpha_recip_x(D, N, unitD, alpha, k1=1, k2=1, return_sum=True):
    n = D.shape[0] # total number of nodes
    
    # force amplitudes is the main part the algorithm
    # calculate amplitude
    x_to_k2 = torch.pow(N, k2) # raise x to power of k2
    F = F * k1/n * torch.where(x_to_k2!= 0, k1 / x_to_k2, torch.zeros_like(x_to_k2)).unsqueeze(-1)
    # get the negative directions, for force
    F = -unitD
    
    # apply amplitudes
    
    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i

        # print("After sum")
        # print("F[2]\n", F[2])

    return F
def test_repulsive_force_recip_x():
    hops = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,2,0]])
    alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=hops, alpha=0.3)

    a = [[1,1,1], [0,0,0], [2,2,2], [-1,-1,-1]]
    a = torch.tensor(a).float()
    D = pairwise_difference(a)
    N = torch.norm(D, dim=-1)

    # Element-wise division with mask
    unitD = torch.zeros_like(D)
    mask = N!=0 # Create a mask for non-zero division
    unitD[mask] = D[mask] / N[mask].unsqueeze(-1)
    t = repulsive_force_recip_x(D, N, unitD)
    print(t)
test_repulsive_force_recip_x()
# %%
"""
D (n,n,d) is the pairwaise difference (force). M[i,j]=P[j]-P[i], from i to j
N (n,n) is norm of each pairwise diff element, i.e x.
unitD (n,n,d) is the unit direction, D/N
alpha_hops (n,n) is the coefficient for each hop-distant neighbor
k1 is amplitude factor: k1*f(x)
k2 is decaying factor factor over distance: f(x/k2)
"""
def attractive_force_ahops(D, N, unitD, alpha_hops, k1=1, k2=1, return_sum=True):
    n = D.shape[0] # total number of nodes
    

    # calculate the amplitude
    F = k1 * alpha_hops * (N/k2)
    
    # finally apply the direction
    F = unitD * F.unsqueeze(-1)

    if(return_sum):
        F = F.sum(axis=1) # sum the i-th row to get all forces to i

        # print("After sum")
        # print("F[2]\n", F[2])

    return F

def test_attractive_force_ahops():
    hops = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,2,0]])
    alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=hops, alpha=0.3)

    a = [[1,1,1], [0,0,0], [2,2,2], [-1,-1,-1]]
    a = torch.tensor(a).float()
    D = pairwise_difference(a)
    N = torch.norm(D, dim=-1)

    # Element-wise division with mask
    unitD = torch.zeros_like(D)
    mask = N!=0 # Create a mask for non-zero division
    unitD[mask] = D[mask] / N[mask].unsqueeze(-1)

    t = attractive_force_ahops(D, N, unitD, torch.tensor(alpha_hops))
    print(t)
test_attractive_force_ahops()
# %%
a = [[1,1,1], [0,0,0], [2,2,2]]
a = torch.tensor(a).float()
D = pairwise_difference(a)
N = torch.norm(D, dim=-1)
# Element-wise division with mask
unitD = torch.zeros_like(D)
mask = N!=0 # Create a mask for non-zero division
unitD[mask] = D[mask] / N[mask].unsqueeze(-1)

x = repulsive_force_exp(D, N, unitD, return_sum=True)
print(x)

x = repulsive_force_recip_x(D, N, unitD, return_sum=True)
# hops = torch.tensor([[0,1,1],[1,0,1], [1,1,0]])
# x = attractive_force_ahops(D, N, hops, alpha=0.3)
print(x)

# %% Get the Cora dataset


from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='../datasets', name='Cora')

data = dataset[0]
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
# %%
import networkx as nx
from torch_geometric.utils import to_networkx
Gx = to_networkx(data, to_undirected=True)
A = nx.to_numpy_array(Gx)
# Gx = to_networkit(data, to_undirected=True)
G = nk.nxadapter.nx2nk(Gx)

# %%
edge_index = data.edge_index.numpy()
print(edge_index.shape)
edge_example = edge_index[:, np.where(edge_index[0]==30)[0]]
# %%
G = nk.Graph(data.num_nodes, directed=False)
print(edge_index.shape[1])
for i in range(edge_index.shape[1]):
    e = edge_index[:,i].T
    G.addEdge(e[0], e[1])
# %%
print(G.numberOfNodes(), G.numberOfEdges())
cc = nk.components.ConnectedComponents(G)
cc.run()
print("number of components ", cc.numberOfComponents())
# %%
n = G.numberOfNodes()
# node degrees
degrees = np.array([G.degree(u) for u in G.iterNodes()])

# get distance between all pairs. How about this one
asps = nk.distance.APSP(G)
asps.run()
# all pair hops distance
hops = asps.getDistances(asarray=True)
# print(hops.shape) # nxn matrix

# find max hops
hops[hops>n]=np.inf
maxhops = max(hops[hops<=n])
print("max hops:", maxhops)

mass = np.array([G.degree(u) for u in G.iterNodes()])
alpha = 0.1

# %%
from scipy.stats import norm
"""To determine the standard deviation of a normal distribution 
in a d-dimensional space such that it contains 97 percent of samples"""
def find_standard_deviation(d, percentage):
    q = norm.ppf(percentage, loc=0, scale=1)  # Quantile function for the standard normal distribution
    std_dev = q / ((-2 * math.log(1 - percentage)) ** (1/2))  # Standard deviation calculation
    return std_dev

d = 3  # Dimensionality of the space
percentage = 0.97  # Desired percentage of samples



# %% all pair euclidian distance.
# first set random positions
d = 6 # a 6 dimensional space
n = G.numberOfNodes() # number of nodes

# nsqrt_3 = np.sqrt(n/np.pi) # std ~ r = sqrt(n/pi)
# Z = np.random.normal(0, nsqrt_3, size=(n,d))


# %%
# Generate tensor of 0/1 values following a normal distribution
size = (1000,2)
mu,std=(0.5, 1)
# print(torch.rand(size))
normal_values = (torch.rand(size)>0.1).float()
# normal_values = normal_values.float()
print(normal_values)
print(torch.sum(normal_values)/torch.numel(normal_values))
# V = torch.bernoulli(normal_values)

# Print the tensor V
# print(V.sum()/size)
# %%
# randomly distribute the initial position of nodes in a normal distribution with
# standard deviation 
std0 = find_standard_deviation(d, percentage=0.67) # 
print(find_standard_deviation(3, percentage=0.67) )
print(find_standard_deviation(4, percentage=0.67) )
print(find_standard_deviation(5, percentage=0.67) )
print(std0)
# now calculate all pairs euclidian distnace


# %%
random_points = np.random.normal(0, std0, size=(n,d)) 
print(f"A matrix of shape {random_points.shape} where each row corresponds to embedding of a node in a {d}-dimensional space.")
def run_forcedirected(Points, mass, hops, alpha, *args, **kwargs):
    outputfilename = 'temp-points.npy'
    logfilename = 'log.txt'

    if('outputfilename' in kwargs):
        outputfilename = kwargs.outputfilename
    if('logfilename' in kwargs):
        logfilename = kwargs.logfilename


    logfile = open(logfilename, "w")

    # alpha_hops = get_alpha_hops(hops, alpha)
    alpha_hops = np.apply_along_axis(get_alpha_hops, axis=1, arr=hops, alpha=alpha)
    alpha_hops = torch.tensor (alpha_hops)

    mass = torch.tensor(mass)
    Points = torch.tensor(Points)
    Points_0 = Points.detach().clone()
    V0 = torch.zeros_like(Points)
    Fa = torch.zeros_like(Points)
    Fr = torch.zeros_like(Points)

    print(Points.size())
    print(Fa.size())
    stderr_change = 0

    max_iterations = 4000
    for _iter in range(max_iterations):
        Points_0 = Points.detach().clone() # save current points
        
        # std_noise = np.exp(-stderr_change)*std0 + np.exp(-stderr_change/100)*std0*0.01 
        std_noise = np.exp(-stderr_change)*std0 + 0.005
        stderr_change += 0.005
        
        # std_noise=0
        # if(_iter < max_iterations*0.9):
        #     std_noise = std0*(1 - _iter/(max_iterations*0.9))
        
        # random force application. apply noise on randomly selected points
        random_select = torch.rand(Points.size())>(_iter/max_iterations)
        random_select = random_select.float()

        relocation_noise = torch.normal(0, std_noise, size=Points.shape)*random_select
        print(f"err-std:{std_noise:.2f} | ones ratio:", torch.sum(random_select)/torch.numel(random_select))
        Points += relocation_noise

        # Calculate forces
        D = pairwise_difference(Points)
        N = torch.norm(D, dim=-1)
        unitD = torch.zeros_like(D)
        mask = N!=0 # Element-wise division with mask
        unitD[mask] = D[mask] / N[mask].unsqueeze(-1)

        # find forces
        Fa = attractive_force_ahops(D, N, unitD, alpha_hops)
        Fr = repulsive_force_recip_x(D, N, unitD, k1=2, k2=2)        
        # Fr = repulsive_force_exp(D, N, unitD, k1=2, k2=2)        
        F = (Fa+Fr)*random_select

        # find acceleration on each point a = F/m
        a = torch.where(mass[:, None] != 0, F / mass[:, None], torch.zeros_like(F))
        # a = np.divide(F, mass[:, None], out=np.zeros_like(Fa), where=mass[:, None]!=0)

        # finally apply relocations
        Points += a

        # make reports
        total_attractive_force = torch.sum(torch.norm(Fa, dim=1))
        total_repulsive_force = torch.sum(torch.norm(Fr, dim=1))
        force_sum = torch.norm(torch.sum(Fa+Fr, dim=0))
        total_relocations = torch.sum(torch.norm(Points - Points_0, dim=1))
        total_reloc_noise = torch.sum(torch.norm(relocation_noise, dim=1))
        
        logstr = f"iter{_iter:4d} | "
        logstr+= f"attr:{total_attractive_force:<9.3f}  "
        logstr+= f"repl:{total_repulsive_force:<9.3f}  "
        logstr+= f"sum:{force_sum:<9.3f}  "
        logstr+= f"{total_attractive_force + total_repulsive_force:9.3f} \n"
        logstr+= f"relocs:{total_relocations:9.3f} "
        logstr+= f"err-relocs:{total_reloc_noise:9.3f} \n"

        print(logstr)
        logfile.write(logstr)
        logfile.flush()

        # save the data
        if((_iter)%100==99):
            # save the file
            with open(outputfilename, 'wb') as f:
                np.save(f, Points.numpy())

    logfile.close()

    with open(outputfilename, 'wb') as f:
        np.save(f, Points.numpy())

    ##########################################################
    ##########################################################
# lim = 100
# run_forcedirected(Z[:lim], mass[:lim], hops[:lim, :lim], alpha=0.3)
run_forcedirected(random_points, mass, hops, alpha=0.3)

if __name__ == "__main__":
    exit()
############################################################
############################################################
# %%
############################################################


# %%
# load the saved points
points = np.load("Points.npy")
# Calculate forces
D = pairwise_difference(torch.Tensor(points))
N = torch.norm(D, dim=-1)

# %%
_hops = hops.copy()
print(hops.shape)

# %%
print(D.min())
print(torch.mean(N), torch.std(N))
print(_hops.shape)
print(_hops[0,1])

print(N.size(), hops.shape)
ht = _hops[_hops<np.inf]
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
import torch
import numpy as np

def find_upper_triangle(matrix):
    n = matrix.shape[0]
    upper_triangle = torch.triu(matrix, diagonal=1)
    upper_values, upper_indices = torch.max(upper_triangle, dim=1)
    min_value, min_index = torch.min(upper_values, dim=0)
    max_value, max_index = torch.max(upper_values, dim=0)
    upper_indices = upper_indices.numpy()
    upper_indices = np.column_stack(np.triu_indices(n, k=1))
    argmin_index = upper_indices[min_index]
    argmax_index = upper_indices[max_index]
    return min_value.item(), max_value.item(), argmin_index, argmax_index

# Example usage
n = 4
matrix = torch.tensor([[0, 8, 3, 4],
                       [5, -1, 7, 1],
                       [9, 10, 0, 12],
                       [13, 14, 15, 0]])

min_value, max_value, argmin_index, argmax_index = find_upper_triangle(matrix)
print("Minimum value:", min_value)
print("Maximum value:", max_value)
print("Argmin index:", argmin_index)
print("Argmax index:", argmax_index)

# %%

mask = torch.triu(torch.ones_like(matrix), diagonal=1)
print(torch.argmin(matrix[mask.bool()]))
upper = matrix[mask.bool()]
min_value, min_index = torch.min(upper, dim=0)
print(min_value, min_index)
print(upper)
# print(mask.bool())

# %%
print(points.shape)

# %%
Z = points.copy()
plt.close("all")
# %%
#Draw
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(Z)
x_pca = pd.DataFrame(x_pca)
x_pca.head()  

# %%
print(list(x_pca.columns))
# %% draw 2d
pca = PCA(n_components=2)
x_pca = pca.fit_transform(Z)
x_pca = pd.DataFrame(x_pca)
# plt.scatter(x_pca.loc[:, 0], x_pca.loc[:, 1], s=np.log(degrees+1))
plt.scatter(x_pca.loc[:, 0], x_pca.loc[:, 1], s=degrees/10)
fig=plt.figure()

# drawGraph_forcedirected(G, x_pca.to_numpy(), Fa=Fa, Fr=Fr, with_labels=False)
drawGraph_forcedirected(G, x_pca.to_numpy(), with_labels=False)

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
print(np.unique(y))
# %%
# Create the figure and 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot for each class
scatter_plots = []
for i, (x, y, z, class_label) in enumerate(plotdata):
    scatter = ax.scatter(x, y, z, c=class_label, cmap='tab10', label=f'Class {i}')
    scatter_plots.append(scatter)

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
# data.y[:10]
print(type(data.y))
# print(len(data.y))
print(len(np.unique(data.y.numpy())))
# %%
a = [[1,2,1,2],[0,0,0,0],[1,2,1,2],[0,0,0,0]]
a = np.array(a)
print("a:\n", a)
b = a+a.T
print("b:\n", b)
c = np.array([1,1,1,1])
print("b-c:\n", b-c)
d = b-c
print(np.linalg.norm(d, axis=1))
print(np.linalg.norm(d, axis=0))
# %%
