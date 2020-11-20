from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from numpy import linalg as LA
from scipy.sparse import csr_matrix
from scipy.io import loadmat, savemat
import math
import random
import sys
import time
np.set_printoptions(threshold=sys.maxsize)


def USPEC(fea, Ks, distance = 'euclidean', p=1000, Knn=5, maxTcutKmIters=100, cntTcutKmReps=3):
    N = fea.shape[0]

    if p > N:
        p = N
    # print warning: off message

    start = time.time()
    RpFea = getRepresentitivesByHibridSelection(fea, p)
    end = time.time()
    print('time for getRpFea: ', end - start)

    cntRepCls = math.floor(math.sqrt(p))

    start = time.time()
    if (distance == 'euclidean'):
        repClsLabel, repClsCenters = litekmeans(RpFea, cntRepCls, MaxIter=20)
    else:
        repClsLabel, repClsCenters = litekmeans(RpFea, cntRepCls, MaxIter=20)
    end = time.time()
    print('time for find center of cluster: ', end - start)

    centerDist = pdist2_fast(fea, repClsCenters, distance);
    # Find the nearest rep-cluster (in RpFea) for each object
    minCenterIdxs = np.argmin(centerDist, axis=1)  # one dim
    # print(minCenterIdxs)
    # clear centerDist
    cntRepCls = repClsCenters.shape[0]
    # print(cntRepCls)

    start = time.time()
    # Then find the nearest representative in the nearest rep-cluster for each object.
    nearestRepInRpFeaIdx = np.zeros((N, 1), dtype=np.int64)  # here is 2 dim
    
    for i in range(cntRepCls):
        # cluster is from 0 to 9 including 9, in matlab 1:10 including 10
        # calculate the min index from fea which is nearest to cluster i to all rep in cluster i
        # nearestRepInRpFeaIdx[np.where(minCenterIdxs==i)]=np.argmin(pdist2_fast(fea[np.where(minCenterIdx==i),:],RpFea[np.where(repClsLabel==i),:],distance),axis=1)
        # need to increase the dim of argmin
        # print(pdist2_fast(fea[np.where(minCenterIdxs==i)[0],:],RpFea[np.where(repClsLabel==i)[0],:],distance).shape)

        tmp = np.argmin(
            pdist2_fast(fea[np.where(minCenterIdxs == i)[0], :], RpFea[np.where(repClsLabel == i)[0], :], distance),
            axis=1)
        # print(tmp.shape)
        # print(nearestRepInRpFeaIdx[np.where(minCenterIdxs==i)].shape)
        nearestRepInRpFeaIdx[np.where(minCenterIdxs == i)] = np.expand_dims(tmp, axis=1)
        # the index of rep equals i
        tmp = np.where(repClsLabel == i)
        # print(np.squeeze(nearestRepInRpFeaIdx[np.where(minCenterIdxs==i)],axis=1).shape)

        # invert offset index to real index of rep
        tmp[0][np.squeeze(nearestRepInRpFeaIdx[np.where(minCenterIdxs == i)], axis=1)]
        nearestRepInRpFeaIdx[np.where(minCenterIdxs == i)] = tmp[0][nearestRepInRpFeaIdx[np.where(minCenterIdxs == i)]]
    # print(nearestRepInRpFeaIdx)
    # the result is 2 dim
    end = time.time()
    print('find the nearest representative in the nearest rep-cluster for each object: ', end - start)

    # For each object, compute its distance to the candidate neighborhood of its nearest representative not need to be
    # in one cluster(in RpFea)
    neighSize = 10 * Knn  # The candidate neighborhood size. K' = knn*10
    RpFeaW = pdist2_fast(RpFea, RpFea, distance)  # distance matrix
    RpFeaKnnIdx = np.argsort(RpFeaW, axis=1)  # too long may

    start = time.time()
    RpFeaKnnIdx = RpFeaKnnIdx[:, 0:neighSize + 1]  # (p,K'+1)
    # same method to nearestRepInRpFeaIdx
    RpFeaKnnDist = np.zeros((N, RpFeaKnnIdx.shape[1]))  # entry_i to K' distances and nearest rc
    
    for i in range(p):
        # print(fea[np.where(nearestRepInRpFeaIdx==i)[0],:].shape)
        RpFeaKnnDist[np.where(nearestRepInRpFeaIdx == i), :] = pdist2_fast(
            fea[np.where(nearestRepInRpFeaIdx == i)[0], :], RpFea[RpFeaKnnIdx[i, :], :], distance)
    
    # get full matrix for each entry with K' nearest reps in indices form
    # select rows based on nearestRepInRpFeaIdx to create N * K' matrix
    RpFeaKnnIdxFull = RpFeaKnnIdx[np.squeeze(nearestRepInRpFeaIdx, axis=1),
                      :]  # entry index corresponding to RpFeaKnnDist
    # print(RpFeaKnnIdxFull)
    end = time.time()
    print('compute its distance to the candidate neighborhood of its nearest representative (in RpFea): ', end - start)

    start = time.time()
    knnDist = np.zeros((N, Knn))
    knnTmpIdx = np.zeros((N, Knn), dtype=np.int64)
    knnIdx = np.zeros((N, Knn), dtype=np.int64)
    for i in range(Knn):
        knnTmpIdx[:, i] = np.argmin(RpFeaKnnDist, axis=1)
        knnDist[:, i] = np.min(RpFeaKnnDist, axis=1)
        rowIdx = np.arange(N)
        RpFeaKnnDist[rowIdx, knnTmpIdx[:, i]] = 1e100  # set the accessed rep with large number
        knnIdx[:, i] = RpFeaKnnIdxFull[
            rowIdx, knnTmpIdx[:, i]]  # mapping the index to rep cluster index which is nearest to the entry
    # print(knnIdx)
    end = time.time()
    print('Get the final KNN according to the candidate neighborhood: ', end - start)

    ## Compute the cross-affinity matrix B for the bipartite graph

    start = time.time()
    if distance == 'euclidean':
        knnMeanDiff = knnDist.mean(axis=None)
        Gsdx = np.exp(-np.square(knnDist**2)/(2*knnMeanDiff**2))

    Gsdx[Gsdx==0]= np.finfo(float).eps
    Gidx = np.arange(N).reshape(N,1)+np.zeros(Knn)
    B = csr_matrix((Gsdx.flatten('F'), (Gidx.flatten('F'), knnIdx.flatten('F'))), shape=(N, p))

    #TODO: extend to Ks as an array
    labels = np.zeros(shape=(N, 1))
    labels[:, 0] = TCut_for_bipartite_graph(B, Ks, maxTcutKmIters, cntTcutKmReps)
    end = time.time()
    print('Compute the cross-affinity matrix B for the bipartite graph: ', end - start)
    return labels


def litekmeans(X, K, MaxIter=100, Replicates=1, Start='random'):
    """
    func:
        partitions the points in the (N,p) matrix X into K clusters. This partition minimizes the sum
        of the within-cluster sums of point-to-cluster-centroid distance.

    arguments:
        'X' - the data matrix, size - (N,p), rows indicate the points, columns indicate the variables.
        'K' - the number of clusters.
        'MaxIter' Optional - Maximum number of iterations allowed.  Default is 100.
        'Replicates' Optional - Number of times to repeat the clustering, each with a new set of initial
            centroids. Default is 1. If the initial centroids are provided, the replicate will be
            automatically set to be 1.
        'Distance' - Distance measure, in P-dimensional space, that KMEANS should minimize with respect to.
            Choices are: {'sqEuclidean'} - Squared Euclidean distance (the default), 'cosine' ignore.
        'Start' - Method used to choose initial cluster centroid positions, sometimes known as "seeds".
            Choices are:
            {'sample'}  - Select K observations from X at random (the default),  'matrix' ignore.
    return:
        'label' -  an N-by-1 vector containing the cluster indices of each point. dim = 1
        'center' - the K cluster centroid locations in the K-by-P matrix center. dim = 2
    """
    km = KMeans(n_clusters=K, init=Start, max_iter=MaxIter).fit(X)
    center = km.cluster_centers_
    label = km.labels_
    return [label, center]


def distEucSq(X, Y):
    Yt = np.transpose(Y)

    D = np.absolute(np.sum(X * X, 1)[:, np.newaxis] + np.sum(Yt * Yt, 0) - 2 * np.dot(X, Yt))
    return D


def pdist2_fast(X, Y, metric):
    """
    func:
        Calculates the distance between sets of vectors.
    argument:
        'X' - matrix (N,d) N is the number of entries in dataset, d is the dimension of data.
        'Y' - matrix (K,d), K is number of rep-clusters.
        'metric' - the way to measure distanct.
    """

    if metric == 'sqeuclidean':
        D = distEucSq(X, Y)
    elif metric == 'euclidean':
        D = np.sqrt(distEucSq(X, Y))
    else:
        D = None
    return D


def getRepresentitivesByHibridSelection(fea, pSize, cntTimes=10):
    N = np.shape(fea)[0]
    bigPSize = cntTimes * pSize
    if pSize > N:
        pSize = N
    if bigPSize > N:
        bigPSize = N

    #idx = random.sample(list(range(N)), bigPSize)
    #selected_fea = fea[idx]
    #random_fea = np.array(selected_fea)
    random_fea = fea[np.random.choice(N, size=bigPSize)]

    kmeans = MiniBatchKMeans(n_clusters=pSize, max_iter=10, tol=1, init='random').fit(random_fea)
    
    return kmeans.cluster_centers_



def TCut_for_bipartite_graph(B, Nseg, maxKmIters=100, cntReps=3):
    Nx, Ny = B.shape
    # print(Nx, Ny)
    if Ny < Nseg:
        sys.exit("Not enough columns!")
    dx = B.sum(axis=1)

    # print(dx.shape)
    dx[dx==0]=1e-10

    helpmat = np.squeeze(np.asarray(1.0/dx))
    # print(helpmat.shape)
    #Dx = np.zeros(shape=(Nx, Nx))
    #np.fill_diagonal(Dx, helpmat)
    Dx = csr_matrix((helpmat, (list(range(helpmat.shape[0])), list(range(helpmat.shape[0])))), shape=(Nx, Nx))
    # print(Dx.shape) # should be Nx*Nx

    Wy = B.transpose().dot(Dx).dot(B)

    # print(Wy.shape)
    ### Computer NCut eigenvectors
    # normalized affinity matrix
    d = Wy.sum(axis=1)
    # print(d.shape)
    helpmat2 = np.squeeze(np.asarray(1.0/np.sqrt(d)))
    #D = np.zeros(shape=(Ny, Ny))
    #np.fill_diagonal(D, helpmat2)
    D = csr_matrix((helpmat2, (list(range(helpmat2.shape[0])), list(range(helpmat2.shape[0])))), shape=(Ny, Ny))
    # print(D.shape)
    nWy = D.dot(Wy).dot(D)
    #nWy = np.matmul(np.matmul(D, Wy), D)
    nWy = (nWy + nWy.transpose())/2

    # computer eigenvectors
    eval, evec = LA.eig(nWy.toarray())
    idx = (-eval).argsort()[1:Nseg]
    Ncut_evec = D.dot(evec[:, idx])

    ### computer the Ncut eigenvectors on the entire bipartite graph (transfer!)
    evec = csr_matrix(Dx.dot(B).dot(Ncut_evec))

    bottom0 = evec.multiply(evec)
    bottom = np.sqrt(bottom0.sum(axis=1)) + 1e-10

    evec = evec.multiply(np.power(bottom, -1))
    kmeans = KMeans(n_clusters=Nseg, max_iter=maxKmIters, n_init=cntReps, init='random').fit(evec)

    return kmeans.labels_


data = loadmat('./USPEC/MATLAB_source_code/data_TB1M.mat')
fea = data['fea']
gt = data['gt']

start = time.time()
labels = USPEC(fea, 2)
end = time.time()

savemat('output/output_TB1M.mat', {'label': labels})
print('total time: ', end - start)
