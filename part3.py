from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg as LA
import math
import sys

def TCut_for_bipartite_graph(B, Nseg, maxKmIters=100, cntReps=3):
    Nx, Ny = B.shape

    if Ny < Nseg:
        sys.exit("Not enough columns!")
    dx = B.sum(axis=1)

    if not np.any(dx):
        dx.fill(1e-10)
    Dx = np.diag(1.0/dx)
    print(Dx.shape) # should be Nx*Nx

    Wy = np.matmul(np.matmul(B.transpose(), Dx), B)

    ### Computer NCut eigenvectors
    # normalized affinity matrix
    d = Wy.sum(axis=1)
    D = np.diag(1.0/math.sqrt(d))
    nWy = np.matmul(np.matmul(D, Wy), D)
    nWy = (nWy + nWy.transpose())/2

    # computer eigenvectors
    eval, evec = LA.eig(nWy)
    idx = (-eval).argsort()[1:Nseg]
    Ncut_evec = np.matmul(D, evec[:, idx])

    ### computer the Ncut eigenvectors on the entire bipartite graph (transfer!)
    evec = np.matmul(np.matmul(Dx, B), Ncut_evec)

    bottom0 = np.multiply(evec, evec)
    bottom = np.sqrt(bottom0.sum(axis=1)) + 1e-10

    evec = evec / bottom
    kmeans = KMeans(n_clusters=Nseg, max_iter=maxKmIters, n_iter=cntReps).fit(evec)

    return kmeans
