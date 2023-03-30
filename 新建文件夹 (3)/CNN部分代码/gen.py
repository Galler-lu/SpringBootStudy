import sys

# import pyod.models.abod
# import pyod.models.knn
# import pyod.models.pca
# import pyod.models.mcd
# import pyod.models.ocsvm
# import pyod.models.lof
import function_mysql
import function_data
import function_test
import numpy as np
import pickle as pk
import os
import matplotlib.pyplot as plt
import pylab as pl
import scipy.signal as signal
import random
import math
import time
import Function01

from scipy.signal import medfilt
from scipy import signal
from sklearn.utils.extmath import safe_sparse_dot
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
# from thundersvm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
import joblib

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import AdaBoostClassifier
#
# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print('开始时间：', time.strftime('%Y-%m-%d %X', time.localtime(time.time())))
    normal_data_list = []  # 初始化正常情况下的数据列表
    sick_data_list = []  # 初始化发病情况下的数据列表
    print('测试样本为全部')

    ############     训练    #############
    flag = 0
    account = ''
    normal_time = ''
    sick_time = ''
    for i in range(17):
        if i == 0:
            # continue
            account = 13101097823
            normal_time = "2020-12-16 10:00:00%2020-12-16 17:00:00&2020-12-16 19:00:00%2020-12-16 23:00:00"
            # normal_time = "2020-12-16 17:00:00%2020-12-16 18:00:00"
            # sick_time = "2020-12-16 17:38:46%2020-12-16 17:57:46"  # 发病前19分钟
            sick_time = "2020-12-16 17:57:47%2020-12-16 18:00:00"  # 发病

        if i == 1:
            continue
        if i == 2:
            continue
        if i == 3:
            continue
            # 王江曼 WJM–18003
            account = 13961327467
            normal_time = "2020-12-24 12:00:00%2020-12-24 22:00:00&2020-12-25 12:00:00%2020-12-25 20:00:00"
            # normal_time = "2020-12-24 22:04:03%2020-12-24 23:04:03"
            # sick_time = "2020-12-24 22:41:34%2020-12-24 23:00:34&2020-12-25 20:19:23%2020-12-25 20:38:43"  # 发病前19分钟
            sick_time = "2020-12-24 23:00:35%2020-12-24 23:04:03&2020-12-25 20:38:44%2020-12-25 20:40:45"  # 发病
        if i == 4:
            continue
        if i == 5:
            continue
            # 陆艳 LY-46529
            account = 15820761208
            normal_time = "2021-01-03 02:00:00%2021-01-03 04:00:00"
            # normal_time = "2021-01-03 04:19:46%2021-01-03 05:19:46"
            sick_time = "2021-01-03 05:19:00%2021-01-03 05:19:46"  # 发病
            # sick_time = "2021-01-03 04:59:59%2021-01-03 05:18:59"  # 发病前19分钟
        if i == 6:
            continue
        if i == 7:
            continue
        if i == 8:
            continue
        if i == 9:
            continue
            account = 17637907651
            # LHR LHR-47347
            normal_time = "2021-02-24 01:00:00%2021-02-24 07:00:00"
            # normal_time = "2021-02-24 07:12:44%2021-02-24 08:12:44"
            sick_time = "2021-02-24 08:11:32%2021-02-24 08:12:44"  # 发病
            # sick_time = "2021-02-24 07:52:31%2021-02-24 08:11:31"  # 发病前19分钟
        if i == 10:
            continue
        if i == 11:
            continue
            # 贺晓丽 HXL-48229
            account = 15805363569
            normal_time = "2021-04-22 02:00:00%2021-04-22 12:00:00"
            # sick_time = "2021-04-22 00:12:45%2021-04-22 00:31:45"  # 发病前19分钟
            sick_time = "2021-04-22 00:31:46%2021-04-22 00:32:54"  # 发病
        if i == 12:
            continue
        if i == 13:
            continue
        if i == 14:
            continue
            # 郭文秀 GWX-48491
            account = 15036067997
            normal_time = "2021-05-30 07:00:00%2021-05-30 23:59:59"
            # normal_time = "2021-05-30 03:40:15%2021-05-30 04:40:15"
            # sick_time = "2021-05-30 04:19:38%2021-05-30 04:38:38"  # 发病前19分钟
            sick_time = "2021-05-30 04:38:39%2021-05-30 04:40:15"  # 发病
        if i == 15:
            continue
        if i == 16:
            continue
            # 常亮 CL-50629
            account = 13889886092
            normal_time = "2021-10-23 03:00:00%2021-10-23 19:00:00"
            # normal_time = "2021-10-23 19:43:01%2021-10-23 20:43:01"
            # sick_time = "2021-10-23 20:22:27%2021-10-23 20:41:27"  # 发病前19分钟
            sick_time = "2021-10-23 20:41:28%2021-10-23 20:43:01"  # 发病

        normal_time_array = normal_time.split("&")
        for item in normal_time_array:
            normal_begin = item.split("%")[0]
            normal_end = item.split("%")[1]
            normal_tuple = function_mysql.conn_mysql(account, normal_begin, normal_end)
            normal_data_list.append(normal_tuple)

        sick_time_array = sick_time.split("&")
        for item in sick_time_array:
            sick_begin = item.split("%")[0]
            sick_end = item.split("%")[1]
            sick_tuple = function_mysql.conn_mysql(account, sick_begin, sick_end)
            sick_data_list.append(sick_tuple)
        ########################    数据处理    #################################
    # print(normal_data_list)
    all_normal_acc = []
    all_sick_acc = []
    for i in range(len(sick_data_list)):
        for j in range(len(sick_data_list[i])):
            mid_sick = medfilt(sick_data_list[i][j][7:-1], kernel_size=5)
            mid_sick = sick_data_list[i][j][7:-1] - mid_sick
            all_sick_acc = all_sick_acc + list(mid_sick)
    all_normal_gsr = []
    all_sick_gsr = []
    all_normal_hrt = []
    all_sick_hrt = []

    all_normal_wrist = []
    all_sick_wrist = []
    for i in range(len(normal_data_list)):
        for j in range(len(normal_data_list[i])):
            all_normal_gsr = all_normal_gsr + [normal_data_list[i][j][1]]
            all_normal_hrt = all_normal_hrt + [normal_data_list[i][j][2]]
            all_normal_wrist = all_normal_wrist + [normal_data_list[i][j][5]]

    for i in range(len(sick_data_list)):
        for j in range(len(sick_data_list[i])):
            all_sick_gsr = all_sick_gsr + [sick_data_list[i][j][1]]
            all_sick_hrt = all_sick_hrt + [sick_data_list[i][j][2]]
            all_sick_wrist = all_sick_wrist + [sick_data_list[i][j][5]]
    for i in range(len(normal_data_list)):
        for j in range(len(normal_data_list[i])):
            mid_normal = medfilt(normal_data_list[i][j][7:-1], kernel_size=5)
            mid_normal = normal_data_list[i][j][7:-1] - mid_normal
            all_normal_acc = all_normal_acc + list(mid_normal)

    # #
    # # ###################      窗口采样    ###############################
    #
    raw_normal_gsr = []
    raw_sick_gsr = []
    raw_normal_hrt = []
    raw_sick_hrt = []
    raw_normal_acc = []
    raw_sick_acc = []

    raw_normal_wrist = []
    raw_sick_wrist = []

    number_count = 0
    raw_sj = 6
    number_of_point_in_each_sample = raw_sj * 20  # 采样窗口大小
    test_sick_acc = []
    for i in range(raw_sj):
        for j in range(i, len(all_sick_acc) - number_of_point_in_each_sample, number_of_point_in_each_sample):
            sick_small_data_batch_acc = [float(x) for x in all_sick_acc[j: j + number_of_point_in_each_sample]]
            number_count += 1
            raw_sick_acc.append(sick_small_data_batch_acc)
            test_sick_acc.append([sick_small_data_batch_acc, [1]])

    number_of_point_in_each_sample = raw_sj  # 采样窗口大小

    test_sick_ghw = []
    raw_sick_all = []

    for i in range(number_of_point_in_each_sample):
        for j in range(i, len(all_sick_gsr) - number_of_point_in_each_sample, number_of_point_in_each_sample):
            sick_small_data_batch_gsr = [float(x) for x in all_sick_gsr[j: j + number_of_point_in_each_sample]]
            sick_small_data_batch_hrt = [float(x) for x in all_sick_hrt[j: j + number_of_point_in_each_sample]]
            sick_small_data_batch_wrist = [float(x) for x in all_sick_wrist[j:j + number_of_point_in_each_sample]]
            raw_sick_gsr.append(sick_small_data_batch_gsr)  # 添加到整体数据集中
            raw_sick_hrt.append(sick_small_data_batch_hrt)
            raw_sick_wrist.append(sick_small_data_batch_wrist)

            test_sick_ghw.append([sick_small_data_batch_gsr, sick_small_data_batch_hrt, sick_small_data_batch_wrist])
            raw_sick_all.append(raw_sick_gsr + raw_sick_hrt + raw_sick_wrist + raw_sick_acc + [1])
            # print(raw_sick_all)

    for i in range(len(test_sick_ghw)):
        test_sick_ghw[i] = test_sick_ghw[i] + test_sick_acc[i]
        print(type(test_sick_ghw[i]))
        # print(np.array(test_sick_ghw[i]))
        print('test_sick_ghw[i]的维度为：', np.array(test_sick_ghw[i]).shape)
        # print(test_sick_ghw[i])
        # print("==" * 10, test_sick_ghw[i][4])
        # print("==" * 10, test_sick_ghw[i][0])
    # print(raw_sick_all)

    # for i in range(0, len(raw_sick_all)):
    #     for j in range(0,len(raw_sick_all[i])):
    #
    #         with open("all_feature_10/1_normal", "a+") as f2:
    #             if j == 5:
    #                 f2.writelines(str(raw_sick_all[i][j]) + "\n")
    #             else:
    #                 f2.writelines(str(raw_sick_all[i][j]) + ",")

    test_normal_acc = []
    test_normal_ghw = []
    test_normal_all = []
    number_count = 0
    number_of_point_in_each_sample = raw_sj * 20
    for i in range(0, len(all_normal_acc) - number_of_point_in_each_sample + 1, number_of_point_in_each_sample):
        normal_small_data_batch_acc = [float(x) for x in all_normal_acc[i: i + number_of_point_in_each_sample]]
        number_count += 1
        test_normal_acc.append([normal_small_data_batch_acc, [0]])
        # print(test_normal_acc)
        # test_normal_all.append([normal_small_data_batch_acc])

        raw_normal_acc.append(normal_small_data_batch_acc)
    number_of_point_in_each_sample = raw_sj
    for i in range(0, len(all_normal_gsr) - number_of_point_in_each_sample + 1, number_of_point_in_each_sample):
        normal_small_data_batch_gsr = [float(x) for x in all_normal_gsr[i: i + number_of_point_in_each_sample]]
        normal_small_data_batch_hrt = [float(x) for x in all_normal_hrt[i: i + number_of_point_in_each_sample]]
        normal_small_data_batch_wrist = [float(x) for x in all_normal_wrist[i:i + number_of_point_in_each_sample]]
        raw_normal_gsr.append(normal_small_data_batch_gsr)  # 添加到整体数据集中
        raw_normal_hrt.append(normal_small_data_batch_hrt)
        raw_normal_wrist.append(normal_small_data_batch_wrist)
        test_normal_ghw.append(
            [normal_small_data_batch_gsr, normal_small_data_batch_hrt, normal_small_data_batch_wrist])
    for i in range(len(test_normal_ghw) - 1):
        test_normal_ghw[i] = test_normal_ghw[i] + test_normal_acc[i]
        # print(test_normal_ghw[i],test_normal_ghw[i].shape())

    test_all = test_normal_ghw + test_sick_ghw

    print('test_all的维度为：', np.array(test_all).shape)

    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, datasets
    import torch.optim as optim


    # dataSet加载数据集
    class DiabetesDataset(Dataset):
        def __init__(self, data):
            # print(type(data))
            # print(data)
            data = np.array(data)
            # xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
            self.len = len(data)
            # print("==" * 10, self.len,type(data),data.shape,data)
            self.x_data = torch.from_numpy(data[:, :, :4])
            # print(self.x_data)

            self.y_data = torch.from_numpy(data[:, :, -1])
            # print(self.y_data)

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.len


    dataset = DiabetesDataset(test_all)
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)


    # 卷积神经网络模型
    class NetModel1(torch.nn.Module):
        def __init__(self):
            super(NetModel1, self).__init__()
            self.conv_ghwa_1 = torch.nn.Conv1d(2, 16, kernel_size=3, padding=1)
            self.conv_ghwa_2 = torch.nn.Conv1d(16, 32, kernel_size=3, padding=1)
            self.relu = torch.nn.ReLU()
            self.averagePooling1 = torch.nn.AvgPool1d(6)  # 32*1
            # self.averagePooling2 = torch.nn.AvgPool1d(9)
            # self.averagePooling3 = torch.nn.AvgPool1d(20)
            self.linear = torch.nn.Linear(6, 2)
            self.conv_ghwa_3 = torch.nn.Conv1d(64, 1, kernel_size=1)
            self.sigm = torch.nn.Sigmoid()

        def forward(self, x, x_acc):
            x = self.relu(self.conv_ghwa_1(x))
            x = self.relu(self.conv_ghwa_2(x))
            # x = self.averagePooling1(x)#平均池化
            # x_acc = self.averagePooling1(self.relu(self.conv_ghwa_1(x_acc)))
            # x_acc = self.averagePooling1(self.relu(self.conv_ghwa_2(x_acc)))
            x_acc = self.relu(self.conv_ghwa_1(x_acc))
            x_acc = self.relu(self.conv_ghwa_2(x_acc))
            # x_acc = self.averagePooling1(x_acc)#平均池化
            print("x为：", x, type(x), x.size())
            print("===" * 20)
            print("x_acc为：", x_acc, type(x_acc), x_acc.size())
            print("===" * 20)
            out = torch.cat((x, x_acc), dim=1)
            print("out为:", out, out.size())
            out = self.conv_ghwa_3(out)
            print("out为:", out, out.size())
            print(self.sigm(self.linear(out)))
            return self.sigm(self.linear(out))
            # x=self.linear(x)
            # x_acc=self.linear(x_acc)

    # print(test_all)
    # """
    # Module 5
    # 对特征集数据进行处理
    # 按照七三原则，切分得到训练集和测试集
    # # """
    # for x in feature_sick_data:  # add label
    #     x.append(1)
    # for x in feature_normal_data:
    #     x.append(0)
    # ##################    选用不同测试集（上是7/3分，下是选择的测试集）   ################################################################
    # test_ill = feature_sick_data[:int(len(feature_sick_data) * 0.3)]
    # test_normal = feature_normal_data[:int(len(test_ill))]
    # train_ill = feature_sick_data[int(len(feature_sick_data) * 0.3):]
    # train_normal = feature_normal_data[int(len(test_ill)): int(len(test_ill) + len(train_ill))]
    # ####################################################################################################
    # test_ill.extend(test_normal)  # merge the ill and normal data
    # train_ill.extend(train_normal)
    # random.shuffle(test_ill)
    # random.shuffle(train_ill)
    # train_data = train_ill
    # test_data = test_ill
    # print(len(test_data[0]), len(train_data[0]))
    # print(len(train_data), len(test_data))
    #
    # # 加载相关数据
    # feature, label = function_data.split_feature_label(train_data)
    # ## print(len(label), float(len(label) - sum(label)), sum(label))
    # print('The ratio between P sample and N sample in Train Set is ', sum(label) / float(len(label) - sum(label)))
    # test_fea, test_la = function_data.split_feature_label(test_data)
    # ## print(len(test_la), float(len(test_la) - sum(test_la)), sum(test_la))
    # print('The ratio between P sample and N sample in Test Set is ', sum(test_la) / float(len(test_la) - sum(test_la)))
    #
    # # ##########################    将发病数据与未发病数据分开    #############################
    # sick_data_border = []
    # normal_data_border = []
    # sick_data_border_test = []
    # normal_data_border_test = []

    # """
    # 将发病数据与未发病数据分开，便于检验边界函数
    # """
    # for k in range(len(label)):
    #     if label[k] == 1:
    #         sick_data_border.append(feature[k])
    #     else:
    #         normal_data_border.append(feature[k])
    #
    # for l in range(len(test_la)):
    #     if test_la[l] == 1:
    #         sick_data_border.append(test_fea[l])
    #         sick_data_border_test.append(test_fea[l])
    #     else:
    #         normal_data_border.append(test_fea[l])
    #         normal_data_border_test.append(test_fea[l])

    # list1=[[[1,1,1,1],[2,2,2,2]]]
    # list2=[[[3,3,3,3],[4,4,4,4]]]
    # print(list1 + list2)
    # list1.extend(list2)
    # print(list1)
