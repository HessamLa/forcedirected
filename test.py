# %%
import torch
import numpy as np
import time

import random
d = 6
N = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if CUDA is available, otherwise use CPU
# %%
N = 10000
def pairwise_difference(P):
    n, d = P.size()
    # Expand dimensions of P to create row-wise repetitions
    P_row = P.unsqueeze(1)    
    # Expand dimensions of P to create column-wise repetitions
    P_col = P.unsqueeze(0)
    # Compute the matrix M
    # print(P_row.size(), P_col.size())
    D = P_col - P_row
    return D

vector = torch.randn(N, d) * 100
X = pairwise_difference(vector)
Xnp = X.numpy()
print(X.size())

def make_random_matrix(N):
    # Generate a random matrix with values in the range 1 to 8
    matrix = np.random.randint(1, 9, size=(N, N))

    # Make the matrix symmetric by setting the lower triangular part equal to the upper triangular part
    matrix = np.triu(matrix) + np.triu(matrix, k=1).T

    # Set the diagonal elements to 0
    np.fill_diagonal(matrix, 0)
    return matrix


ttorch_wmask = 0
tnumpy_wmask = 0
ttorch_nomask = 0
tnumpy_nomask = 0
for i in range(5):
    # TORCH NO MASK ############################################
    hops = make_random_matrix(N)
    hops = torch.from_numpy(hops).to(device)
    alpha = 0.387
    
    tstart = time.time()
    a = torch.pow(alpha, hops - 1) * (hops != 0).to(torch.float32)
    Y = X*a.unsqueeze(-1)
    tend = time.time()
    # print(Y.size(), X.size(), a.size())
    print("torch, no mask:", tend-tstart)
    ttorch_nomask += tend-tstart
    
    #############################################

    # NUMPY NO MASK ############################################
    hops = make_random_matrix(N).astype(float)
    alpha = 0.387
    
    tstart = time.time()
    a = np.power(alpha, hops-1, out=np.zeros_like(hops), where=hops!=0)
    Y = a[:,:,np.newaxis] * Xnp
    tend = time.time()
    
    # print(Y.shape, X.shape, a.shape)
    print("numpy, no mask:", tend-tstart)    

    tnumpy_nomask += tend-tstart
    #############################################

    #############################################
    hops = [random.randint(0, 3) for _ in range(N)]
    hops = make_random_matrix(N)
    hops = torch.from_numpy(hops).to(device)
    alpha = 0.387
    
    tstart = time.time()
    mask = hops != 0
    a = torch.zeros_like(hops, dtype=torch.float32, device=device)
    a[mask] = torch.pow(alpha, hops[mask] - 1)
    Y = X*a.unsqueeze(-1)
    tend = time.time()
    
    # print(Y.size(), X.size(), a.size())
    print("torch, with mask:", tend-tstart)
    ttorch_wmask += tend-tstart
    #############################################

    # NUMPY W MASK ############################################
    hops = make_random_matrix(N).astype(float)
    alpha = 0.387
    
    tstart = time.time()
    mask = hops != 0
    a = np.zeros_like(hops)
    a[mask] = np.power(alpha, hops[mask]-1)
    Y = a[:,:,np.newaxis] * Xnp
    tend = time.time()

    # print(Y.shape, X.shape, a.shape)
    print("numpy, with mask:", tend-tstart)    
    tnumpy_wmask += tend-tstart
    #############################################

print(f"time torch w/ mask: {ttorch_wmask}")
print(f"time torch no mask: {ttorch_nomask}")
print(f"time numpy w/ mask: {tnumpy_wmask}")
print(f"time numpy no mask: {tnumpy_nomask}")
print("")

# %%
def test_func(f1, f2):
    f1()
    f2()

def do_f1(x, y=8):
    print(x, y)

f1 = lambda:do_f1(10)
f2 = lambda:do_f1(20, y=19)
test_func(f1=f1, f2=f2)

# %%
from math import exp
# solve f(x) for x
k1=10
k2=0.9
alpha=0.3
h=4
def f(x, k1=k1, k2=k2, alpha=alpha, h=h):
    return alpha**(h-1)*x - h*k1 * exp(-x/k2)

def df(x, k1=k1, k2=k2, alpha=alpha, h=h):
    return alpha**(h-1) + h*k1*k2 * exp(-x/k2)


def newton_raphson(x0, tol=1e-6, max_iter=100):
    x = x0
    iter_count = 0
    
    fx = lambda: f(x, k1=k1, k2=k2, alpha=alpha, h=h)
    dfx = lambda: df(x, k1=k1, k2=k2, alpha=alpha, h=h)
    
    while abs(fx()) > tol and iter_count < max_iter:
        x = x - fx() / dfx()
        iter_count += 1
    
    if abs(fx()) <= tol:
        return x
    else:
        return None

# Solve the equation
for h in range(1,20):
    solution = newton_raphson(x0=1.0)
    if solution is not None:
        print(f"h={h} x={solution:.4f}")
    else:
        print("No solution found within the specified tolerance or maximum iterations.")