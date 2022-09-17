import numpy as np


class k_means_serial(object):
    def __init__(self, n_clusters, max_iter):
        self.num_of_cluster = n_clusters
        self.max_iter = max_iter

    def min_cluster_dist(self, clusters, x):
        return np.argmin([self.euc_dist(x, c) for c in clusters])

    def euc_dist(self, a, b):
        return np.sqrt(((a - b) ** 2).sum())

    def points_clustering(self, blobs):     # clustering
        self.cluster_prediction = [self.min_cluster_dist(self.cluster_centers_, x) for x in blobs]
        ind = []
        for j in range(self.num_of_cluster):
            current_cluster = []
            for i, k in enumerate(self.cluster_prediction):
                if k == j:
                    current_cluster.append(i)
            ind.append(current_cluster)
        points_for_cluster = [blobs[i] for i in ind]
        return points_for_cluster

    def random_centroids(self, blobs):
        centroid = np.random.permutation(blobs.shape[0])[:self.num_of_cluster]
        return blobs[centroid]

    def fit(self, blobs):
        self.cluster_centers_ = self.random_centroids(blobs)
        for i in range(self.max_iter):
            X_by_cluster = self.points_clustering(blobs)

            new_cl_centers = [c.sum(axis=0) / len(c) for c in X_by_cluster] # new clsuter centers
            new_cl_centers = [arr.tolist() for arr in new_cl_centers]
            old_centers = self.cluster_centers_

            if np.all(new_cl_centers == old_centers): # check convergence
                self.number_of_iter = i
                break
            else:
                self.cluster_centers_ = new_cl_centers # keep iter
        return self


class kmeans_parallel(object):
    def __init__(self, n_clusters):
        self.num_of_cluster = n_clusters


