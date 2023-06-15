# %%
%reload_ext autoreload
%autoreload 2
import networkit as nk
%matplotlib inline
import matplotlib.pyplot as plt
import math
import numpy as np

import networkx as nx
import sklearn as sk
import pandas as pd
from utils import drawGraph_forcedirected

# %%
"""
Simple enumeration class to list supported file types. Possible values:
networkit.graphio.Format.DOT
networkit.graphio.Format.EdgeList
networkit.graphio.Format.EdgeListCommaOne
networkit.graphio.Format.EdgeListSpaceZero
networkit.graphio.Format.EdgeListSpaceOne
networkit.graphio.Format.EdgeListTabZero
networkit.graphio.Format.EdgeListTabOne
networkit.graphio.Format.GraphML
networkit.graphio.Format.GraphToolBinary
networkit.graphio.Format.GraphViz
networkit.graphio.Format.GEXF
networkit.graphio.Format.GML
networkit.graphio.Format.KONEC
networkit.graphio.Format.LFR
networkit.graphio.Format.METIS
networkit.graphio.Format.NetworkitBinary
networkit.graphio.Format.SNAP
"""
filepath="../datasets/snap/ego-facebook/facebook_combined.txt"
G = nk.readGraph(filepath, nk.graphio.Format.EdgeListSpaceZero)
# %%
print(G.numberOfNodes(), G.numberOfEdges())
# get adjacency matrix

#%% get distance between all pairs. How about this one
asps = nk.distance.APSP(G)
asps.run()
# %% all pair hops distance
hops = asps.getDistances(asarray=True)
print(hops.shape) # nxn matrix

# node degrees
degrees = np.array([G.degree(u) for u in G.iterNodes()])

# %%
h = hops[0, :]
total=0
for i in range(int(h.max())+1):
    print(i, len(h[h==i]))
    total+=len(h[h==i])
print("total", total)

h.shape
# %% all pair euclidian distance.
# first set random positions
d = 4 # a 6 dimensional space
n = G.numberOfNodes() # number of nodes
Z = np.random.rand(n,d)
print(f"A matrix of shape {Z.shape} where each row corresponds to embedding of a node in a {d}-dimensional space.")
# now calculate all pairs euclidian distnace

# # method 1
# b = a.reshape(a.shape[0], 1, a.shape[1])
# D = np.sqrt(np.einsum('ijk, ijk->ij', a-b, a-b))
# print(D.shape)
# print(D[:5, :5])

# method 2
from sklearn.metrics.pairwise import euclidean_distances
D = euclidean_distances(Z,Z)
# print(D.shape)
# print(D[:5, :5])


# %%
def repulsive_force_degreemas_edist(Z0, zu, mass, return_sum=True):
    # z = Z0[u,:]
    n = Z0.shape[0] # total number of nodes
    dZ = zu-Z0
    nZ = np.linalg.norm(dZ, axis=1) # optimize this
    unitZ = np.divide(dZ.T, nZ, out=np.zeros_like(dZ.T), where=nZ!=0).T

    # F = mass/n*np.divide(dZ.T, nZ**3, out=np.zeros_like(dZ.T), where=nZ**3!=0) # apply transpose to use the broadcast feature
    F = 10/n*unitZ.T*np.exp(-nZ) # apply transpose to use the broadcast feature
    F = F.T
    #print(F.shape)
    # F = F.sum(axis=1)
    if(return_sum):
        F = F.sum(axis=0)
    return F

for u in G.iterNodes():
    repulsive_force_degreemas_edist(points, points[u,:], degrees)
    break

# print(np.ones(a.shape[0], out=np.zeros_like(a), where=a==1))
# %%
a = np.array([[1,1,1],[3,4,1]])
b = np.array([[1,0,1],[0,1,0]])
print(b*a)
b = np.array([1,0,1])
print(b*a)


c = np.unique(a, return_counts=True)
divided_arr = np.divide(1, c[1][np.searchsorted(c[0], a)])
print(divided_arr)
# %%
"""F_attractive = \Sum_{h=1}^{\inf}\Sum_{v \in N^{h}(u) \alpha^{h-1}degree(v)(z_u - z_v)/||z_u - z_v||}
only connected component graph can be passed to this function
"""
def attractive_force_degmass_hopdist(Z0, zu, mass, hops, alpha, return_sum=True):
    c = np.unique(hops, return_counts=True) # number of nodes at i-hop distance
    nhop_1 = np.divide(1, c[1][np.searchsorted(c[0], hops)]) 
    # example [1,1,2,3,3] -> [0.5, 0.5, 1, 0.5, 0.5] 
    # each element is divided by its corresponding unique count

    # exponent = np.divide(alpha, hops**2, out=np.zeros_like(hops), where=hops!=0)
    # a = np.exp(exponent, out=np.zeros_like(hops), where=hops!=0) # \alpha^{h-1})
    # a = np.exp(-alpha*hops**2, out=np.zeros_like(hops)) # \exp^{-\alpha*hops})
    
    a = np.power(alpha, hops-1, out=np.zeros_like(hops), where=hops!=0)

    # a = np.power(alpha, hops, out=np.zeros_like(hops), where=hops!=np.inf) # \alpha^{h-1}
    # print(np.stack([a,hops]).T)
    # print("")

    dZ = Z0-zu                      # (z_u - z_v)
    nZ = np.linalg.norm(dZ, axis=1) #||z_u - z_v||
    unitZ = np.divide(dZ.T, nZ, out=np.zeros_like(dZ.T), where=nZ!=0).T # apply transpose to use the broadcast feature
    # print(dZ.shape, nZ.shape)
    # print(unitZ.shape)
    # print(dZ)
    
    # F = a * degrees * unitZ.T
    # F = a * mass * unitZ.T * nZ.T
    F = a * nhop_1 * dZ.T /2
    F = F.T
    if(return_sum):
        F = F.sum(axis=0)
    # print(F.shape)
    return F

alpha=0.00003
drawGraph_forcedirected(G, points, distance_scale, alpha, mass)

# for u in G.iterNodes():
#     attractive_force_degmass_hopdist(Z, Z[u,:], degrees, hops[u,:], 0.3)
#     break


# %%
n = 20
# while(True):
#     erg = nk.generators.ErdosRenyiGenerator(n, 1/np.sqrt(n))
#     G = erg.generate()
#     cc = nk.components.ConnectedComponents(G)
#     cc.run()
#     if(cc.numberOfComponents() == 1): break
n = G.numberOfNodes() # number of nodes
asps = nk.distance.APSP(G)
asps.run()
hops = asps.getDistances(asarray=True)
# node degrees
degrees = np.array([G.degree(u) for u in G.iterNodes()])

distance_scale=n
# points = np.random.rand(n,2)*distance_scale
points = Z

mass = np.array([G.degree(u) for u in G.iterNodes()])
alpha = 0.3

Fa = np.zeros_like(points)
Fr = np.zeros_like(points)
for u in G.iterNodes():
    # Fa = attractive_force_degmass_hopdist(points, points[u], mass, hops[u,:], alpha=alpha)
    Fa[u] = attractive_force_degmass_hopdist(points, points[u], mass, hops[u,:], alpha=alpha)
    Fr[u] = repulsive_force_degreemas_edist(points, points[u], mass)

drawGraph_forcedirected(G, points, Fa=Fa, Fr=Fr, distance_scale=1)

change = 0

# %%
for _ in range(1000):
    a = Fa+Fr
    b = np.divide(Fa+Fr, degrees[:, None], out=np.zeros_like(Fa), where=degrees[:, None]!=0)
    
    std = np.exp(-change)*100
    change += 0.1

    e = np.random.normal(0, std, size=points.shape)
    points += e

    points += np.divide(Fa+Fr, degrees[:, None], out=np.zeros_like(Fa), where=degrees[:, None]!=0)
    total_attractive_force=0
    total_repulsive_force=0
    alpha = 0.01
    ka, kr = (1, 1)
    k = (np.exp(change)+9)/10
    for u in G.iterNodes():
        # Fa = attractive_force_degmass_hopdist(points, points[u], mass, hops[u,:], alpha=alpha)
        Fa[u] = ka*attractive_force_degmass_hopdist(points, points[u], mass, hops[u,:], alpha=alpha)
        Fr[u] = kr*repulsive_force_degreemas_edist(points, points[u], mass)
        total_attractive_force += np.linalg.norm(Fa[u,:])
        total_repulsive_force += np.linalg.norm(Fr[u,:])
    print("Total attractive forces:", total_attractive_force)
    print("Total repulsive forces:", total_repulsive_force)    
    print("all :", total_attractive_force + total_repulsive_force)
# drawGraph_forcedirected(G, points, Fa, Fr, distance_scale=1)
# %%
drawGraph_forcedirected(G, points)

# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(points)
x_pca = pd.DataFrame(x_pca)
x_pca.head()  

# %%
print(list(x_pca.columns))
# %% draw 2d
pca = PCA(n_components=2)
x_pca = pca.fit_transform(points)
x_pca = pd.DataFrame(x_pca)
# plt.scatter(x_pca.loc[:, 0], x_pca.loc[:, 1], s=np.log(degrees+1))
plt.scatter(x_pca.loc[:, 0], x_pca.loc[:, 1], s=degrees/10)
# fig=plt.figure()
# drawGraph_forcedirected(G, x_pca.to_numpy(), Fa, Fr, distance_scale,with_labels=False, draw_attractions=False, draw_repulsions=False)

# %% draw 3d
pca = PCA(n_components=3)
x_pca = pca.fit_transform(points)
x_pca = pd.DataFrame(x_pca)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_pca.loc[:, 0], x_pca.loc[:, 1], x_pca.loc[:, 2], s=np.log(degrees+1)/10)
# drawGraph_forcedirected(G, x_pca.to_numpy(), Fa, Fr, distance_scale,with_labels=False, draw_attractions=False, draw_repulsions=False)

# %%
# save the file
with open('positions.npy', 'wb') as f:
    np.save(f, points)
# %%
# load the file
with open('positions.npy', 'rb') as f:
    points = np.load(f)
# %%

temperature = 0
alpha = 0.1
mass = np.array([G.degree(u) for u in G.iterNodes()])
v0 = np.zeros_like(points)
Fa = np.zeros_like(points)
Fr = np.zeros_like(points)
for i in range(1):
    points0 = points.copy()
    total_attractive_force = 0
    total_repulsive_force = 0
    # jitter the points according to temperature
    temperature -= 0.001
    e = np.random.normal(0, np.exp(temperature), size=points0.shape)
    points0 += e

    for u in G.iterNodes():
        # Fa = attractive_force_degmass_hopdist(points, points[u], mass, hops[u,:], alpha=alpha)
        Fa[u,:] = ka*attractive_force_degmass_hopdist(points0, points0[u], mass, hops[u,:], alpha=alpha)
        Fr[u,:] = kr*repulsive_force_degreemas_edist(points0, points0[u], mass)
        # points[u] = points0[u] + (Fa + Fr)/degrees[u]
        total_attractive_force += np.linalg.norm(Fa[u,:])
        total_repulsive_force += np.linalg.norm(Fr[u,:])
    points += np.divide(Fa+Fr, degrees[:, None], out=np.zeros_like(Fa), where=degrees[:, None]!=0)
    v0=points-points0
    print("temperature", temperature)
    print("Total attractive forces:", total_attractive_force)
    print("Total repulsive forces:", total_repulsive_force)
    print("")
    # if(i%20==0):
    drawGraph_forcedirected(G, points, title=f"iter {i+1}")
# drawGraph_forcedirected(G, points, distance_scale, alpha, mass, title=f"iter {i+1}")
# %%
drawGraph_forcedirected(G, points, title=f"iter {i+1}")

# %% calculate distance errors
def distance_err_u (Z0, zu, hops):
    dZ = Z0-zu                      # (z_u - z_v)
    nZ = np.linalg.norm(dZ, axis=1) #||z_u - z_v||
    print(nZ.shape, hops.shape)
    dist_err = np.divide(nZ, hops, out=np.zeros_like(nZ), where=hops!=0)
    print(dist_err.shape)
    return dist_err.sum()

for u in G.iterNodes():
    derr = distance_err_u(Z, Z[u,:], hops[u,:])
    print(derr)
    break

# %%
def distance_err_all (Z, hops):
    assert Z.shape[0] == hops.shape[0]
    dist_matrix = euclidean_distances(Z,Z)
    dist_errs = np.divide(dist_matrix, hops, out=np.zeros_like(dist_matrix), where=hops!=0)
    # get 1-hop distances
    idx = hops==1
    one_hops_mean, one_hops_std = (dist_errs[idx].mean(), dist_errs[idx].std())
    idx = hops==2
    two_hops_mean, two_hops_std = (dist_errs[idx].mean(), dist_errs[idx].std())
    
    means = dist_errs.mean(axis=1)
    stds = dist_errs.std(axis=1)
    return (means,stds, one_hops_mean, one_hops_std, two_hops_mean, two_hops_std)

print(Z.shape, hops.shape)
(means,stds, one_hops_mean, one_hops_std, two_hops_mean, two_hops_std) = distance_err_all(Z, hops)
print(f"all     : {means.sum():12.2f} {stds.sum():12.2f}")
print(f"one-hops: {one_hops_mean:12.2f} {one_hops_std:12.2f}")
print(f"two-hops: {two_hops_mean:12.2f} {two_hops_std:12.2f}")

# %%
# %%
import pandas as pd
from sklearn.cluster import KMeans
# %%
from IPython import display
def plotkmeans(X, n_clusters, title=""):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df = pd.DataFrame(X, columns=[f"feat{i+1:02d}" for i in range(Z.shape[-1])])

    df['cluster']= kmeans.fit_predict(df)

    # get centroids
    centroids = kmeans.cluster_centers_
    cen_x = [i[0] for i in centroids] 
    cen_y = [i[1] for i in centroids]
    
    # print(centroids)
    ## add to df
    df['cen_x'] = df.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
    df['cen_y'] = df.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})

    # # define and map colors
    # colors = ['#DF2020', '#81DF20', '#2095DF']
    # df['c'] = df.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})

    # use default color maps
    df['c'] = df['cluster'].map(plt.get_cmap('Dark2'))

    _ = plt.figure()
    plt.scatter(df.feat01, df.feat02, c=df.c, alpha = 0.6, s=10)
    plt.title(f"{title} (feat01 x feat02)")
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.show()
    # _ = plt.figure()
    # plt.scatter(df.feat03, df.feat04, c=df.c, alpha = 0.6, s=10)
    # _ = plt.figure()
    # plt.scatter(df.feat05, df.feat06, c=df.c, alpha = 0.6, s=10)

plotkmeans(Z, 15, title="test")
# %% apply the forces
# # anealing parameters
# init_temp = 100
# d_temp = 0.1

plotkmeans(Z, 15, title=f"iteration {0}")
logfile = open("log.txt", "w")

alpha_attractive = 0.1
for i in range(400):
    logstr = f"iter {i+1}\n"
    print(logstr)

    Z0 = Z.copy()
    # # apply simulated anealing
    # temp = init_temp/(i+1)
    # aneal = math.exp(-d_temp/temp)

    for u in G.iterNodes():
        Fa = attractive_force_degmass_hopdist(Z0, Z0[u,:], degrees, hops[u,:], alpha_attractive)
        Fr = repulsive_force_degreemas_edist(Z0, Z0[u,:], degrees)
        Z[u] = Z0[u] + (Fa - Fr)/degrees[u]

    (means,stds, one_hops_mean, one_hops_std, two_hops_mean, two_hops_std) = distance_err_all(Z, hops)
    eZ = np.linalg.norm(Z-Z0, axis=1)
    # print(f"anealing factor: {aneal}")
    logstr += f"change  : {eZ.mean():12.2f} {eZ.std():12.2f}, {eZ.max()}, {eZ.min()}\n"
    logstr += f"all     : {means.sum():12.2f} {stds.sum():12.2f}\n"
    logstr += f"one-hops: {one_hops_mean:12.2f} {one_hops_std:12.2f}\n"
    logstr += f"two-hops: {two_hops_mean:12.2f} {two_hops_std:12.2f}\n"
    logfile.write(logstr)
    logfile.flush()

    #draw clustering
    if(i%5==0):
        plotkmeans(Z, 15, title=f"iteration {i+1}")

logfile.close()
plotkmeans(Z, 15, title=f"iteration last({i})")

# %%
print(f"anealing factor: {aneal}")
print(temp, i)
print(math.exp(-1*1/100))
e = Z-Z0
print(e.shape)
ez = np.linalg.norm(Z-Z0, axis=1)
print(ez.shape)
print(ez.mean(), ez.std())
# %%
print(Z.shape)
mu = Z.mean(axis=0)
std = Z.std(axis=0)
print(np.linalg.norm(mu), np.linalg.norm(std))
# %%
print(Z.shape, Z0.shape, Fa.shape, Fr.shape, degrees.shape)


# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(Z)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
# %%
from sklearn.decomposition import PCA
x_coord = [vec[6] for vec in Z]
y_coord = [vec[7] for vec in Z]

plt.clf()
plt.scatter(x_coord, y_coord)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("2 Dimensional Representation of Graph Embeddings on Randomly Generated Networks")
plt.show()


    
# %%

print(plt.get_cmap('tab20'))
cm = plt.get_cmap('Dark2')
maps={}
for i in range(len(df.cluster.unique())):
    maps[i]=cm(i)

print(maps)
df['c'] = df.cluster.map(cm)