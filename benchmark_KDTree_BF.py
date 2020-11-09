# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import os
import struct
from scipy.spatial import KDTree
import tqdm


def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32).T


def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1
    # bf search
    points = read_velodyne_bin('000000.bin').transpose()

    process_scipy = True
    process_bf = True

    if process_scipy:

        sci_kdtree = KDTree(points, leaf_size)
        test_time = 100
        cost_scipy = time.time()
        for ii in tqdm.tqdm(range(test_time)):
            sci_kdtree = KDTree(points, leaf_size)
        cost_scipy = time.time() - cost_scipy
        print('scipy.KDTree build cost: {0} ms'.format(cost_scipy*1000/test_time))

        cost_scipy = time.time()
        for ii in tqdm.tqdm(range(points.shape[0])):
            res = sci_kdtree.query(points[ii, :], k)
        cost_scipy = time.time() - cost_scipy
        print('scipy.KDTree query cost: {0} ms'.format(cost_scipy*1000))
    if process_bf:
        cost_bf = time.time()
        for ii in tqdm.tqdm(range(points.shape[0])):
            diff = np.linalg.norm(np.expand_dims(points[ii,:], 0) - points, axis=1)
            for jj in range(k):
                # bf 对每个点计算距离， 对knn， 由于k比较小，排序划不来， 每次把最小的距离置为无穷大
                knn_one_result = np.argmin(diff)
                diff[knn_one_result] = np.inf
        cost_bf = time.time() - cost_bf
        print('Numpy brute-force query cost: {0} ms'.format(cost_bf*1000))



if __name__ == '__main__':
    main()
