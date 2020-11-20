from sklearn.cluster import KMeans
import numpy as np
import random

def getRepresentitivesByHibridSelection(fea, pSize, cntTimes = 10):
    N = np.shape(fea)[0]
    bigPSize = cntTimes * pSize
    if pSize > N:
        pSize = N
    if bigPSize > N:
        bigPSize = N
    
    selected_fea = random.sample([x for x in fea], k=bigPSize)
    random_fea = np.array(selected_fea)

    kmeans = KMeans(n_clusters=pSize).fit(random_fea)
    return kmeans.cluster_centers_


B = np.array([[1, 2, 2, 3], [2, 5, 0, 0], [1, 8, 3, 8], [3, 1, 2, 2], [3, 1, 0, 1]])
RpFea = getRepresentitivesByHibridSelection(B, 2)
print(RpFea)