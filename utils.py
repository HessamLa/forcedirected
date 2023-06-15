# %%
import matplotlib.pyplot as plt
import networkit as nk

def drawGraph_forcedirected(G, points, Fa=None, Fr=None, distance_scale=1, with_labels=False, title=""):
    n=points.shape[0]

    positions = {i:points[i,:2] for i in range(n)} # get first two columns
    nk.viztasks.drawGraph(G, pos = positions, with_labels=with_labels)

    for i in range(n):
        if(Fa is not None):
            plt.arrow(points[i,0], points[i,1], Fa[i,0], Fa[i,1], fc='g')
        if(Fr is not None):
            plt.arrow(points[i,0], points[i,1], Fr[i,0], Fr[i,1], fc='r')

    plt.scatter(points[:,0], points[:,1])
    plt.title(title)
    plt.show()