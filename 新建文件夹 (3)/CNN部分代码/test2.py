import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import Kmeans算法.kmeans1
import Function01
import pandas as pd
import Function01

if __name__ == '__main__':
    import pandas as pd

    # data = pd.read_excel("5人大发作/发病期/91特征重复采样.xlsx")
    # iris_centrodis, iris_dataSey = Kmeans算法.kmeans1.KMeans(data,3)
    # print(iris_centrodis)
    # print(iris_dataSey.head())
    # print(type(data))
    # print(data.T)
    # print(data)
    # feature=data.T
    feature=pd.read_excel("5人大发作/发病期/91特征重复采样转置.xlsx")
    # feature.to_excel("5人大发作/发病期/91特征重复采样转置.xlsx")
    data_centrodis, data_result = Kmeans算法.kmeans1.KMeans(feature, 4)
    data_result.to_excel("5人大发作/发病期/91特征重复采样分组分为4簇.xlsx")
    # df = pd.read_excel("0号病人/测试/皮肤电42个特征.xlsx")
    # for i in range(40,42):
    #     plt.subplot(4, 1, i-39)
    #     plt.plot(df.iloc[:, i])
    # plt.show()
#     feature_normal_10 = Function01.readData("11号病人_独立样本T检验/premorbid_5_EDA", 30)
#     # feature_normal_10=np.array(feature_normal_10)
#     # feature_normal_10[np.isnan(feature_normal_10)] = 0
#     # # 标准化，服从正态分布
#     # transfer = StandardScaler()
#     # feature_normal_10 = transfer.fit_transform(feature_normal_10)
#     # print(np.var(feature_normal_10))
#     # print(np.mean(feature_normal_10))
#
#     for i in range(0, len(feature_normal_10)):
#         for j in range(0, len(feature_normal_10[i])):
#             if j==29:
#                 with open("11号病人_独立样本T检验/", "a+") as f2:
#                     f2.writelines(str(feature_normal_10[i][j]) + "\n")
#             else:
#                 with open("11号病人_独立样本T检验/", "a+") as f2:
#                     f2.writelines(str(feature_normal_10[i][j]) + ",")
#     df1=pd.read_csv("部分病人样本非参数检验/normal_EDA")
#     df1=pd.read_excel("部分病人显著性检验/发病前期/心率数据/心率整体数据.xlsx")
#
#     # df1=df1.sample(n=1200)
#     df1=np.random.choice()
#     print(df1)
#     #
#     #
#     df1.to_excel("部分病人归一化测试/premorbid_heart.xlsx")

#
# all_normal_gsr = Function01.readData("部分病人检验/发病前期/皮肤电数据/皮肤电整体数据除去异常值.txt", 1)
# list = []
# for i in range(len(all_normal_gsr)):
#     for j in range(len(all_normal_gsr[i])):
#         list.append(int(all_normal_gsr[i][j]))
# all_normal_gsr = list
# df1 = np.random.choice(all_normal_gsr, size=2000, replace=False)
# pd.DataFrame(df1).to_excel("部分病人检验/发病前期/皮肤电数据/皮肤电抽样数据.xlsx")
#
#
# all_normal_hrt = Function01.readData("部分病人检验/发病前期/心率数据/心率整体数据除去异常值.txt", 1)
# list = []
# for i in range(len(all_normal_hrt)):
#     for j in range(len(all_normal_hrt[i])):
#         list.append(int(all_normal_hrt[i][j]))
# all_normal_hrt = list
# df1 = np.random.choice(all_normal_hrt, size=2000, replace=False)
# pd.DataFrame(df1).to_excel("部分病人检验/发病前期/心率数据/心率抽样数据.xlsx")


# all_normal_gsr = Function01.readData("部分病人检验/发病间期/皮肤电数据/皮肤电整体数据除去异常值.txt", 1)
# list = []
# for i in range(len(all_normal_gsr)):
#     for j in range(len(all_normal_gsr[i])):
#         list.append(int(all_normal_gsr[i][j]))
# all_normal_gsr = list
# df1 = np.random.choice(all_normal_gsr, size=15000, replace=False)
# pd.DataFrame(df1).to_excel("部分病人检验/发病间期/皮肤电数据/皮肤电抽样数据1.xlsx")


# all_normal_hrt = Function01.readData("部分病人检验/发病间期/心率数据/心率整体数据除去异常值.txt", 1)
# list = []
# for i in range(len(all_normal_hrt)):
#     for j in range(len(all_normal_hrt[i])):
#         list.append(int(all_normal_hrt[i][j]))
# all_normal_hrt = list
# df1 = np.random.choice(all_normal_hrt, size=15000, replace=False)
# pd.DataFrame(df1).to_excel("部分病人检验/发病间期/心率数据/心率抽样数据1.xlsx")


# data1=np.array(df1)
# data1.to
# df1=pd.read_excel("14号病人独立样本T检验/1.xlsx")
# df2=pd.read_excel("14号病人独立样本T检验/2.xlsx")
# all_normal_gsr1 = Function01.readData("部分病人显著性检验/发病间期/心率数据/心率除去异常值的抽样数据.txt", 1)
# list = []
# for i in range(len(all_normal_gsr1)):
#     for j in range(len(all_normal_gsr1[i])):
#         list.append(int(all_normal_gsr1[i][j]))
# all_normal_gsr1 = list
# all_normal_gsr2 = Function01.readData("部分病人显著性检验/发病前期/心率数据/心率除去异常值的抽样数据.txt", 1)
# list = []
# for i in range(len(all_normal_gsr2)):
#     for j in range(len(all_normal_gsr2[i])):
#         list.append(int(all_normal_gsr2[i][j]))
# all_normal_gsr2 = list
# # print(len(all_normal_gsr2))
# # print(all_normal_gsr2.shape)
# all_normal_gsr1 = np.array(all_normal_gsr1).reshape(-1, 1)
# all_normal_gsr2 = np.array(all_normal_gsr2).reshape(-1, 1)
# print(all_normal_gsr1.shape)
# print(all_normal_gsr2.shape)
# # print(all_normal_gsr1)
#
# scaler1 = MinMaxScaler()
# data1 = scaler1.fit_transform(all_normal_gsr1)
# scaler2 = MinMaxScaler()
# data2 = scaler2.fit_transform(all_normal_gsr2)
# # scaler2 = StandardScaler()
# # data2=scaler2.fit_transform(df2)
# data1 = pd.DataFrame(data1)
# data2 = pd.DataFrame(data2)
# # # data1=list(data1)
# # print(data1)
# # data2=list(data2)
# # print(type(data1))
# print(data1.shape)
# print(data2.shape)
# data1.to_excel("部分病人显著性检验/发病间期/心率数据/心率除去异常值归一化的抽样数据.xlsx")
# data2.to_excel("部分病人显著性检验/发病前期/心率数据/心率除去异常值归一化的抽样数据.xlsx")
# #
# # from sklearn.metrics.pairwise import cosine_similarity
# #
# # print(cosine_similarity(np.array(data1).reshape(1, -1), np.array(data2).reshape(1, -1)))
# # # # from scipy.stats import pearsonr
# # # #
# # # # print(pearsonr(data1, data2))
# # # # print(cosine_similarity(data1, data2))
# # # # print(data1)
# # # data1.to_excel("测试包/1_normal_Heart.xlsx")
# # # data2.to_excel("测试包/1_premorbid_Heart.xlsx")
# #
# # # arr1=np.array([[3,4,10,7],[8,5,7,1]])
# # # arr2=np.array([[9,20,30,21],[40,50,49,10]])
# # # print(arr1.std())
# # # print(arr2.std())
# # # # list1=[[arr1.std(),arr2.std()],[arr1.std(),arr2.std()]]
# # # # # list1=np.array(list1).reshape(-1,1)
# # # # list1=np.array(list1)
# # # # print(list1)
# # # scaler=MinMaxScaler()
# # # print(scaler.fit_transform(arr1))
# # # print(scaler.fit_transform(arr2))
# # # # print(scaler.fit_transform(list1))
# # # # # a = np.arange(16).reshape(4, 4)
# # # # print(a)
# # # # print(np.diff(a,n=2))
# from sklearn.preprocessing import MinMaxScaler
#
# data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
# # data2=[[-10,20],[-5,60],[0,100],[10,180]]
# # import pandas as pd
# # from sklearn.metrics.pairwise import cosine_similarity
# #
# # print(cosine_similarity(data, data2))
# # print("***************")
# # print(pd.DataFrame(data))
# # print(pd.DataFrame(data2))
# # 实现归一化
# scaler = MinMaxScaler()  # 实例化
# scaler = scaler.fit(data)  # fit，在这里本质是生成min(x)和max(x)
# result = scaler.transform(data)  # 通过接口导出结果
# # print(scaler.fit_transform(data2))
# print(result)
#
# text=[1,2,3,4,5,6,7,8,9]
# list=np.random.choice(text,size=5,replace=False)
# print(list.tolist())
# print(type(list.tolist()))
