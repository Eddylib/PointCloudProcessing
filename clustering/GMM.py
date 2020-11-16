# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random, math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from itertools import cycle, islice

plt.style.use('seaborn')


class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        # miu, pi, sigma
        self.centers = None
        self.pi = None
        self.sigma = None
        self.tolerance_ = 0.00001

    def multi_gauss_distribte(self, x, miu, sigma):
        resized_x = np.reshape(x, (-1, 1))
        resized_miu = np.reshape(miu, (-1, 1))
        D = resized_x.shape[0]
        det_sigma = np.sqrt(np.linalg.det(sigma))
        ret = 1 / ((2 * np.pi) ** (D / 2)) * 1 / det_sigma * np.exp(
            -0.5 * np.matmul(np.matmul((resized_x - resized_miu).transpose(), np.linalg.inv(sigma)), (resized_x - resized_miu)))
        return ret

    def check_converge(self, new_centers):
        mean_distance = np.sum(np.linalg.norm(self.centers - new_centers, axis=1))
        return mean_distance < self.tolerance_

    def fit(self, data):
        # 作业3
        # 屏蔽开始
        data_num = data.shape[0]
        data_dim = data.shape[1]

        self.centers = []
        self.sigma = []

        # 初始中心要距离足够远！
        data_norm = np.linalg.norm(data, axis=1)
        idx = np.argmax(data_norm)
        self.centers.append(data[idx,:])
        self.sigma.append(np.eye(data_dim))
        for ii in range(self.n_clusters-1):
            all_distance_sum = np.zeros((data_num))
            for jj in range(ii+1):
                existing_center = self.centers[jj]
                all_distance_sum += np.linalg.norm(existing_center - data, axis=1)
            new_idx = np.argmax(all_distance_sum)
            self.centers.append(data[new_idx, :])
            self.sigma.append(np.eye(data_dim))
        self.centers = np.vstack(self.centers)
        # 计算初始聚类中心的协方差矩阵、每个类别的pi
        probobility = np.zeros((data_num, self.n_clusters), dtype=np.float)
        for nn in range(data_num):
            for kk in range(self.n_clusters):
                probobility[nn, kk] = self.multi_gauss_distribte(data[nn, :], self.centers[kk, :], self.sigma[kk])
        rnk = np.zeros((data_num, self.n_clusters), dtype=np.float)
        for kk in range(self.n_clusters):
            rnk[:, kk] = probobility[:, kk] / np.sum(probobility, axis=1)
        Nk = np.sum(rnk, axis=0)
        for kk in range(self.n_clusters):
            self.sigma[kk] = 1 / Nk[kk] * (
                np.matmul((rnk[:, kk].reshape((-1, 1)) * (data - self.centers[kk])).transpose(),
                          data - self.centers[kk]))
        Nk = np.sum(rnk, axis=0)
        self.pi = Nk/data_num

        converge = False
        ii = 0
        for ii in range(self.max_iter):
            # E step
            probobility = np.zeros((data_num, self.n_clusters), dtype=np.float)
            for nn in range(data_num):
                for kk in range(self.n_clusters):
                    probobility[nn, kk] = self.multi_gauss_distribte(data[nn, :], self.centers[kk, :], self.sigma[kk])
            rnk = np.zeros((data_num, self.n_clusters), dtype=np.float)
            for kk in range(self.n_clusters):
                rnk[:, kk] = self.pi[kk] * probobility[:, kk] / np.sum(self.pi[:] * probobility, axis=1)

            Nk = np.sum(rnk, axis=0)

            # M setp
            new_centers = np.zeros((self.n_clusters, data_dim))
            new_sigmas = []
            new_pi = np.zeros((self.n_clusters))
            sigularity = False
            for kk in range(self.n_clusters):
                new_centers[kk, :] = 1 / Nk[kk] * np.sum(rnk[:, kk].reshape((-1, 1)) * data, axis=0)
                new_sigma = 1 / Nk[kk] * (
                    np.matmul((rnk[:, kk].reshape((-1, 1)) * (data - new_centers[kk])).transpose(),
                              data - new_centers[kk]))
                new_pi = Nk / data_num
                if (np.linalg.det(new_sigma) < 0.1):
                    sigularity = True
                    new_centers[kk, :] = data[np.random.randint(0, data_num), :]
                    new_sigma = np.eye(data_dim) * 100
                new_sigmas.append(new_sigma)
            if (sigularity):
                new_pi = np.ones(self.n_clusters) / self.n_clusters

            if (self.check_converge(new_centers)):
                converge = True
            self.centers = new_centers
            self.sigma = new_sigmas
            self.pi = new_pi

            if (converge):
                break
        print('converge iterate: {}'.format(ii))

    def predict(self, data):
        ret = []
        # 屏蔽开始
        for ii in range(data.shape[0]):
            prob = np.zeros(self.n_clusters)
            for kk in range(self.n_clusters):
                prob[kk] = self.multi_gauss_distribte(data[ii, :], self.centers[kk], self.sigma[kk])
            ret.append(np.argmax(prob))

        # 屏蔽结束
        return ret

    def draw(self, ax, colors):
        plt.plot(self.centers[:,0], self.centers[:,1], 'r*')
        lumbdas, vectors = np.linalg.eig(self.sigma)
        for ii in range(self.n_clusters):
            xy = self.centers[ii]
            max_idx = np.argmax(lumbdas[ii])
            min_idx = np.argmin(lumbdas[ii])

            width = lumbdas[ii][max_idx]*2
            height = lumbdas[ii][min_idx]*2
            vector = vectors[ii][:,max_idx]
            angle = (math.atan2(vector[1], vector[0]))*180/np.pi
            ell = Ellipse(xy, width=width,height=height, angle=angle, color=colors[ii], alpha=0.1)
            ax.add_artist(ell)
        pass


# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 40, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 60, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 100, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    # plt.figure(figsize=(10, 8))
    # plt.axis([-10, 15, -5, 15])
    # plt.scatter(X1[:, 0], X1[:, 1], s=5)
    # plt.scatter(X2[:, 0], X2[:, 1], s=5)
    # plt.scatter(X3[:, 0], X3[:, 1], s=5)
    # plt.show()
    return X


if __name__ == '__main__':
    # 生成数据
    # true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    # true_Var = [[1, 3], [2, 2], [1, 2]]
    # X = generate_X(true_Mu, true_Var)
    n_samples = 1500
    from sklearn import cluster, datasets, mixture
    X, label = datasets.make_blobs(n_samples=n_samples, random_state=8)

    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    X = X_aniso

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(cat) + 1))))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], color=colors[cat])
    gmm.draw(ax, colors)
    plt.show()
