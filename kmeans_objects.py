import numpy as np
from multiprocessing import Pool


class k_means_serial(object):
    def __init__(self, n_clusters, max_iter):
        self.num_of_cluster = n_clusters
        self.max_iter = max_iter

    def min_cluster_dist(self, clusters, x):
        return np.argmin([self.euc_dist(x, c) for c in clusters])

    def euc_dist(self, a, b):
        return np.sqrt(((a - b) ** 2).sum())

    def points_clustering(self, blobs):     # clustering
        self.cluster_prediction = [self.min_cluster_dist(self.cluster_centers, x) for x in blobs]
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
        self.cluster_centers = self.random_centroids(blobs)
        for i in range(self.max_iter):
            X_by_cluster = self.points_clustering(blobs)

            new_cl_centers = [c.sum(axis=0) / len(c) for c in X_by_cluster] # new clsuter centers
            new_cl_centers = [arr.tolist() for arr in new_cl_centers]
            old_centers = self.cluster_centers

            if np.all(new_cl_centers == old_centers): # check convergence
                self.number_of_iter = i
                break
            else:
                self.cluster_centers = new_cl_centers # keep iter
        return self


class k_means_parallel(k_means_serial):
    def __init__(self, n_clusters, max_iter, num_of_cores):
        super().__init__(n_clusters, max_iter)
        self.num_cores = num_of_cores

    def shuffle(self, list_in, n):
        temp = np.random.permutation(list_in)
        result = [temp[i::n] for i in range(n)]
        return result

    def fit(self, blobs):
        self.cluster_centers = self.random_centroids(blobs)
        for i in range(self.max_iter):

            splitted_data = self.shuffle(blobs, self.num_cores)

            pool = Pool()
            result = pool.map(self.points_clustering, splitted_data)
            pool.close()
            pool.join()
            # splitted e riunito

            points_for_cluster = []
            for k in range(0, self.num_of_cluster):
                points = []
                for pool in range(0, self.num_cores):
                    tmp = result[pool][k].tolist()
                    points = sum([points, tmp], [])
                points_for_cluster.append(np.array(points))

            new_cl_centers = [k.sum(axis=0) / len(k) for k in points_for_cluster]
            new_cl_centers = [np.array(arr) for arr in new_cl_centers]
            old_cl_centers = self.cluster_centers
            old_cl_centers = [np.array(arr) for arr in old_cl_centers]
            if all([np.allclose(x, y) for x, y in zip(old_cl_centers, new_cl_centers)]):
                self.number_of_iter = i
                break
            else:
                self.cluster_centers = new_cl_centers
        self.number_of_iter = i
        return self
