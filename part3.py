from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg as LA
import sys

def TCut_for_bipartite_graph(B, Nseg, maxKmIters=100, cntReps=3):
    Nx, Ny = B.shape
    # print(Nx, Ny)
    if Ny < Nseg:
        sys.exit("Not enough columns!")
    dx = B.sum(axis=1)

    # print(dx.shape)
    if not np.any(dx):
        dx.fill(1e-10)
    helpmat = np.squeeze(np.asarray(1.0/dx))
    # print(helpmat.shape)
    Dx = np.diag(helpmat)
    # print(Dx.shape) # should be Nx*Nx

    Wy = np.matmul(np.matmul(B.transpose(), Dx), B)

    # print(Wy.shape)
    ### Computer NCut eigenvectors
    # normalized affinity matrix
    d = Wy.sum(axis=1)
    # print(d.shape)
    helpmat2 = np.squeeze(np.asarray(1.0/np.sqrt(d)))
    D = np.diag(helpmat2)
    # print(D.shape)
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
    kmeans = KMeans(n_clusters=Nseg, max_iter=maxKmIters, n_init=cntReps).fit(evec)

    return kmeans.labels_


B = np.matrix([[1, 2, 2, 3], [2, 5, 0, 0], [1, 8, 3, 8], [3, 1, 2, 2], [3, 1, 0, 1]])
labels = TCut_for_bipartite_graph(B, 2)
print(labels)
