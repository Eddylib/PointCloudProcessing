import numpy as np
from numpy import *
import pylab
import random, math

from sklearn import cluster, datasets, mixture
from scipy.spatial import cKDTree, KDTree
from scipy.sparse.linalg import eigs as sparse_eig
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from itertools import cycle, islice

from KMeans import K_Means


# plt.style.use('seaborn')


class Spectral(object):
    def __init__(self):
        self.tolerance_ = 0.00001

    # def distance(self, eular_dis):
    #     return 1 / (eular_dis + 0.000001)
    def distance(self, eular_dis):
        return np.exp(-eular_dis)

    def confirm_k(self, value, sort_idx):
        sum_diff = value[sort_idx[0]] - value[sort_idx[1]]
        sorted_value = value[sort_idx]
        prev = sorted_value[0:-1]
        last = sorted_value[1:]
        diff = last-prev
        mean_diff = np.mean(diff[0:5])
        kk = 1
        for kk in range(1, value.shape[0]):
            curr_diff = sorted_value[kk] - sorted_value[kk-1]
            if (curr_diff > mean_diff):
                break
            sum_diff += curr_diff
        return kk

    def fit(self, data):
        data_num = data.shape[0]
        W = np.zeros((data_num, data_num), dtype=np.float)
        D = np.zeros((data_num, data_num), dtype=np.float)
        Dinv = np.zeros((data_num, data_num), dtype=np.float)
        self.kdtree = KDTree(data)
        for ii in range(data_num):
            eular_dis, idx = self.kdtree.query(data[ii, :], k=max(int(data_num / 20), 10))
            distance_all = self.distance(eular_dis)
            W[ii, idx] = distance_all
            W[ii, ii] = 0
        W = np.sqrt(W * W.transpose())
        for ii in range(data_num):
            D[ii, ii] = np.sum(W[ii, :])
            if (D[ii, ii] > 0.0001):
                Dinv[ii, ii] = 1 / D[ii, ii]
            else:
                Dinv[ii, ii] = 1 / 0.0001

        # Lrw = np.matmul(Dinv, D-W)
        Lrw = np.eye(data_num) - np.matmul(Dinv, W)
        # Lrw = D-W
        value, vector = np.linalg.eig(Lrw)
        sort_idx = np.argsort(value)

        k_means_k = self.confirm_k(value, sort_idx)
        # k_means_k = 2
        print('k is evaluated as {}'.format(k_means_k))

        print('idx:', sort_idx[0:k_means_k], 'lambda', value[sort_idx[0:k_means_k]])
        k_means_data = vector[:, sort_idx[0:k_means_k]]

        self.k_means_manager = K_Means(k_means_k)
        self.k_means_manager.fit(k_means_data)
        self.spectral_result = np.array(self.k_means_manager.predict(k_means_data))
        # plt.imshow(W, vmin=0, vmax=100)
        # plt.show()
        # plt.plot(k_means_data[:,0], k_means_data[:,1],'r.')
        # plt.plot(value[sort_idx], 'r.')
        # plt.show()
        # exit(0)

    def predict(self, data):
        ret = []
        # convert to spectral data
        for ii in range(data.shape[0]):
            distance, idx = self.kdtree.query(data[ii, :], k=1)
            ret.append(self.spectral_result[idx])
            # print(spec_data)
        return ret


if __name__ == '__main__':
    # 生成数据
    n_samples = 1500
    X, label = datasets.make_circles(n_samples=n_samples, factor=.5,
                                     noise=.05)
    # random_state = 170
    # X, label = datasets.make_blobs(n_samples=n_samples,
    #                                cluster_std=[1.0, 2.5, 0.5],
    #                                random_state=random_state)
    # plt.plot(X[:,0],X[:,1], 'r.')
    # plt.show()

    spec = Spectral()
    spec.fit(X)
    cat = spec.predict(X)

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(cat) + 1))))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], color=colors[cat])
    plt.show()
