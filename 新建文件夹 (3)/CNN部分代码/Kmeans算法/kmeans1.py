import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
import tensorflow as tf


def dist(arr1, arr2):
    # 采用了欧氏距离
    return np.sqrt(np.sum(np.power((arr1 - arr2), 2), axis=1))


def randCent(dataSet, k):
    n = dataSet.shape[1]
    data_min = dataSet.iloc[:, :n - 1].min()
    data_max = dataSet.iloc[:, :n - 1].max()
    data_cent = np.random.uniform(data_min, data_max, (k, n - 1))
    return data_cent


def KMeans(dataSet, k, distMeans=dist, createCent=randCent):
    m, n = dataSet.shape
    centroids = createCent(dataSet, k)
    clusterAssment = np.zeros((m, 3))
    clusterAssment[:, 0] = np.inf
    clusterAssment[:, 1:3] = -1
    result_set = pd.concat([dataSet, pd.DataFrame(clusterAssment)], axis=1, ignore_index=True)
    clusterChange = True
    while clusterChange:
        clusterChange = False
        for i in range(m):
            #每个样本点与质心计算距离
            dist = distMeans(dataSet.iloc[i, :n - 1].values, centroids)
            result_set.iloc[i, n] = dist.min()
            result_set.iloc[i, n + 1] = np.where(dist == dist.min())[0]
        clusterChange = not (result_set.iloc[:, -1] == result_set.iloc[:, -2]).all()
        if clusterChange:
            cent_df = result_set.groupby(n + 1).mean()
            centroids = cent_df.iloc[:, :n - 1].values
            result_set.iloc[:, -1] = result_set.iloc[:, -2]
    return centroids, result_set


# if __name__ == '__main__':
#     iris = pd.read_csv("iris_training.csv", header=None)
#     # print(iris.head())
#     # print(iris.shape)
#     # print(iris.drop(iris.iloc[0]))
#     # # print(iris)
#     # print(randCent(iris, 3))
#     iris_centrodis, iris_dataSey = KMeans(iris, 3)
#     print(iris_centrodis)
#     print(iris_dataSey.head())
