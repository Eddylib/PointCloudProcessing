# 文件功能： 实现 K-Means 算法

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, islice


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.centers = None

    def check_converge(self, new_centers):
        mean_distance = np.sum(np.linalg.norm(self.centers - new_centers, axis=1))
        return mean_distance < self.tolerance_

    def fit(self, data):
        # 作业1
        # 屏蔽开始

        data_num = data.shape[0]
        data_dim = data.shape[1]

        # 初始中心要距离足够远！
        self.centers = []
        data_norm = np.linalg.norm(data, axis=1)
        idx = np.argmax(data_norm)
        self.centers.append(data[idx,:])
        for ii in range(self.k_-1):
            all_distance_sum = np.zeros((data_num))
            for jj in range(ii+1):
                existing_center = self.centers[jj]
                all_distance_sum += np.linalg.norm(existing_center - data, axis=1)
            new_idx = np.argmax(all_distance_sum)
            self.centers.append(data[new_idx, :])
        self.centers = np.vstack(self.centers)
        # plt.plot(self.centers[:,0], self.centers[:,1], 'rx')

        converge = False
        ii = 0
        for ii in range(self.max_iter_):

            distances = np.zeros((data_num, self.k_), dtype=np.float)
            for kk in range(self.k_):
                tmp_sub = data - self.centers[kk]
                distances[:, kk] = np.linalg.norm(tmp_sub, axis=1)

            new_label = np.argmin(distances, axis=1)
            new_centers = np.zeros((self.k_, data_dim))

            for kk in range(self.k_):
                new_centers[kk, :] = np.mean(data[new_label == kk, :], axis=0)

            if (self.check_converge(new_centers)):
                converge = True
            self.centers = new_centers
            if (converge):
                break
            pass
        print('converge iterate: {}'.format(ii))
        # plt.plot(self.centers[:,0], self.centers[:,1], 'b*')
        # 屏蔽结束

    pass

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        for ii in range(p_datas.shape[0]):
            distances = np.linalg.norm(p_datas[ii] - self.centers, axis=1)
            result.append(np.argmin(distances))
        # 屏蔽结束
        return result


# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    return X


if __name__ == '__main__':
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    k_means = K_Means(n_clusters=3)
    k_means.fit(X)

    cat = k_means.predict(X)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(cat) + 1))))

    plt.scatter(X[:, 0], X[:, 1], color=colors[cat])
    # plt.legend(['initial center',  'final center'])
    plt.show()
