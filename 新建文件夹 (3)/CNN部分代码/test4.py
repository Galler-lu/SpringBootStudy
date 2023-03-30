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
    for i in range(19):
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
            # continue
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
            # 删除此段
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
            # continue
            account = 17637907651
            # LHR LHR-47347
            normal_time = "2021-02-24 01:00:00%2021-02-24 07:00:00"
            # normal_time = "2021-02-24 07:12:44%2021-02-24 08:12:44"
            sick_time = "2021-02-24 08:11:32%2021-02-24 08:12:44"  # 发病
            # sick_time = "2021-02-24 07:52:31%2021-02-24 08:11:31"  # 发病前19分钟
        if i == 10:
            continue
        if i == 11:
            # continue
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
            # continue
            # 郭文秀 GWX-48491
            account = 15036067997
            normal_time = "2021-05-30 07:00:00%2021-05-30 23:59:59"
            # normal_time = "2021-05-30 03:40:15%2021-05-30 04:40:15"
            # sick_time = "2021-05-30 04:19:38%2021-05-30 04:38:38"  # 发病前19分钟
            sick_time = "2021-05-30 04:38:39%2021-05-30 04:40:15"  # 发病
        if i == 15:
            continue
        if i == 16:
            # continue
            # 常亮 CL-50629
            account = 13889886092
            normal_time = "2021-10-23 03:00:00%2021-10-23 19:00:00"
            # normal_time = "2021-10-23 19:43:01%2021-10-23 20:43:01"
            # sick_time = "2021-10-23 20:22:27%2021-10-23 20:41:27"  # 发病前19分钟
            sick_time = "2021-10-23 20:41:28%2021-10-23 20:43:01"  # 发病
        if i == 17:
            # 作为测试集
            continue
            account = 15975021597
            normal_time = "2021-01-09 00:00:00%2021-01-09 15:00:00"
            sick_time = "2021-01-09 15:47:16%2021-01-09 15:48:19"
        if i == 18:
            # 作为测试集
            continue
            account = 13930692558
            normal_time = "2021-01-24 00:00:00%2021-01-24 21:00:00"
            sick_time = "2021-01-24 22:59:04%2021-01-24 23:02:18&2021-01-24 21:49:41%2021-01-24 21:50:30"

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
            sick_small_data_batch_acc=sick_small_data_batch_acc+[1]
            # test_sick_acc.append([sick_small_data_batch_acc, [1]])
            # test_sick_acc.extend([sick_small_data_batch_acc, [1]])
            test_sick_acc.append(sick_small_data_batch_acc)

    number_of_point_in_each_sample = raw_sj  # 采样窗口大小

    print('发病加速度的长度为：', len(test_sick_acc))
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
            # test_sick_ghw = test_sick_ghw + [
            #     sick_small_data_batch_gsr + sick_small_data_batch_hrt + sick_small_data_batch_wrist + [1]]
            test_sick_ghw = test_sick_ghw + [
                sick_small_data_batch_gsr + sick_small_data_batch_hrt + sick_small_data_batch_wrist]

    print('发病其他的长度为：', len(test_sick_ghw))

    for i in range(len(test_sick_ghw)):
        test_sick_ghw[i] = test_sick_ghw[i] + test_sick_acc[i]
        # print(type(test_sick_ghw[i]))
        # print('test_sick_ghw[i]的维度为：', np.array(test_sick_ghw[i]).shape)
        # if i % 100 == 99:
        #     print(test_sick_ghw[i])

    test_normal_acc = []
    test_normal_ghw = []
    test_normal_all = []
    number_count = 0
    number_of_point_in_each_sample = raw_sj * 20
    for i in range(0, len(all_normal_acc) - number_of_point_in_each_sample + 1, number_of_point_in_each_sample):
        normal_small_data_batch_acc = [float(x) for x in all_normal_acc[i: i + number_of_point_in_each_sample]]
        number_count += 1
        normal_small_data_batch_acc=normal_small_data_batch_acc+[0]
        # test_normal_acc.append([normal_small_data_batch_acc, [0]])
        # test_normal_acc.extend([normal_small_data_batch_acc, [0]])
        test_normal_acc.append(normal_small_data_batch_acc)

        raw_normal_acc.append(normal_small_data_batch_acc)
    number_of_point_in_each_sample = raw_sj
    for i in range(0, len(all_normal_gsr) - number_of_point_in_each_sample + 1, number_of_point_in_each_sample):
        normal_small_data_batch_gsr = [float(x) for x in all_normal_gsr[i: i + number_of_point_in_each_sample]]
        normal_small_data_batch_hrt = [float(x) for x in all_normal_hrt[i: i + number_of_point_in_each_sample]]
        normal_small_data_batch_wrist = [float(x) for x in all_normal_wrist[i:i + number_of_point_in_each_sample]]
        raw_normal_gsr.append(normal_small_data_batch_gsr)  # 添加到整体数据集中
        raw_normal_hrt.append(normal_small_data_batch_hrt)
        raw_normal_wrist.append(normal_small_data_batch_wrist)

        # test_normal_ghw = test_normal_ghw + [
        #     normal_small_data_batch_gsr + normal_small_data_batch_hrt + normal_small_data_batch_wrist + [0]]
        test_normal_ghw = test_normal_ghw + [
            normal_small_data_batch_gsr + normal_small_data_batch_hrt + normal_small_data_batch_wrist]

    for i in range(len(test_normal_ghw)):
        test_normal_ghw[i] = test_normal_ghw[i] + test_normal_acc[i]
        if i == 100:
            print(test_normal_ghw[i])

    # test_all = test_normal_ghw + test_sick_ghw
    # print(test_all)

    # print('test_all的维度为：', np.array(test_all).shape)
    test_all = test_normal_ghw
    import xlwt
    import pandas as pd

    rows = test_all

    # workbook = xlwt.Workbook()
    # sheet = workbook.add_sheet("Sheet")
    #
    # for i in range(len(rows)):
    #     for j in range(len(rows[i])):
    #         sheet.write(i, j, rows[i][j])
    #
    # workbook.save("People_7_Normal.xlsx")
    dataframe = pd.DataFrame(data=rows)
    dataframe.to_excel(excel_writer='People_7_Normal.xlsx', index=False, header=False)

    test_all_sick = test_sick_ghw
    import xlwt

    rows = test_all_sick

    # workbook = xlwt.Workbook()
    # sheet = workbook.add_sheet("Sheet")
    #
    # for i in range(len(rows)):
    #     for j in range(len(rows[i])):
    #         sheet.write(i, j, rows[i][j])
    #
    # workbook.save("People_7_Sick.xlsx")
    dataframe = pd.DataFrame(data=rows)
    dataframe.to_excel(excel_writer='People_7_Sick.xlsx', index=False, header=False)

    # import numpy as np
    # import matplotlib.pyplot as plt
    # import torch
    # import torch.nn.functional as F
    # from torch.utils.data import Dataset, DataLoader
    # from torchvision import transforms, datasets
    # import torch.optim as optim
    # import xlwt
    # import openpyxl
    #
    #
    # # dataSet加载数据集
    # class DiabetesDataset(Dataset):
    #     def __init__(self, filepath):
    #         xy = np.array(filepath)
    #         self.len = xy.shape[0]
    #         temp_list = []
    #         for i in range(len(xy)):
    #             temp_list.append([list(xy[i, :6]), list(xy[i, 6:12]), list(xy[i, 12:18])])
    #         print(temp_list)
    #         self.x_data = torch.from_numpy(np.array(temp_list))
    #         print(self.x_data.size())
    #         self.y_data = torch.from_numpy(xy[:, [-1]])
    #         print(self.y_data.size())
    #
    #     def __getitem__(self, index):
    #         return self.x_data[index], self.y_data[index]
    #
    #     def __len__(self):
    #         return self.len
    #
    #
    # my_list = []  # 按行存放Excel表中数据
    # wb = openpyxl.load_workbook('test1.xlsx')
    # ws = wb['Sheet1']
    # maxrows = ws.max_row  # 获取最大行
    # for i in range(maxrows - 1):
    #     temp_list = []
    #     for each in ws.iter_cols(min_row=2):
    #         temp_list.append(each[i].value)
    #     my_list.append(temp_list)
    #
    # dataset = DiabetesDataset(my_list)
    # train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)
    #
    #
    # # 卷积神经网络模型
    # class NetModel1(torch.nn.Module):
    #     def __init__(self):
    #         super(NetModel1, self).__init__()
    #         self.conv_ghwa_1 = torch.nn.Conv1d(3, 16, kernel_size=3, padding=1)
    #         self.conv_ghwa_2 = torch.nn.Conv1d(16, 32, kernel_size=3, padding=1)
    #         self.relu = torch.nn.ReLU()
    #         self.averagePooling1 = torch.nn.AvgPool1d(6)  # 32*1
    #         self.linear = torch.nn.Linear(6, 2)
    #         self.sigm = torch.nn.Sigmoid()
    #
    #     def forward(self, x):
    #         x = self.relu(self.conv_ghwa_1(x))
    #         print('第一个x,size()为：',x.size())
    #         x = self.relu(self.conv_ghwa_2(x))
    #         print('第二个x,size()为：', x.size())
    #         return self.sigm(self.linear(x))
    #
    #
    # model = NetModel1()
    #
    # cirterion = torch.nn.CrossEntropyLoss(size_average=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #
    # # for epoch in range(1000):
    # #     for i, data in enumerate(train_loader, 0):
    # #         inputs, labels = data
    # #         y_pred = model(inputs.float())
    # #         print(y_pred)
    # #         loss = cirterion(y_pred, labels)
    # #         print(epoch, i, loss.item())
    # #         optimizer.zero_grad()
    # #         loss.backward()
    # #         optimizer.step()
    # def train(epoch):
    #     running_loss = 0.0
    #     for batch_idx, data in enumerate(train_loader, 0):
    #         inputs, target = data
    #         optimizer.zero_grad()
    #         # forward + backward + update
    #         outputs = model(inputs.float())
    #         loss = cirterion(outputs, target)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #         if batch_idx % 300 == 299:
    #             print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
    #             running_loss = 0.0
    #
    # train(1000)
