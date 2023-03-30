import sys

import pyod.models.abod
import pyod.models.knn
import pyod.models.pca
import pyod.models.mcd
import pyod.models.ocsvm
import pyod.models.lof
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

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    print('开始时间：', time.strftime('%Y-%m-%d %X', time.localtime(time.time())))
    ######################################################################################
    normal_data_list = []  # 初始化正常情况下的数据列表
    sick_data_list = []  # 初始化发病情况下的数据列表
    normal_testdata_list = []  # 初始化正常情况下的数据列表
    sick_testdata_list = []  # 初始化发病情况下的数据列表
    print('测试样本为全部')
    ############     训练    #############

    account = ''
    normal_time = ''
    sick_time = ''
    for i in range(18):
        # if i == x1:
        #     continue
        if i == 0:
            # 田颖 TY–46316
            account = 13101097823
            normal_time = "2020-12-16 10:00:00%2020-12-16 17:00:00&2020-12-16 19:00:00%2020-12-16 23:00:00"
            # normal_time = "2020-12-16 10:57:47%2020-12-16 17:56:00&2020-12-16 18:02:00%2020-12-16 18:30:00"   #
            sick_time = "2020-12-16 17:57:47%2020-12-16 18:00:00"
        if i == 1:
            continue
            # 孙奕辉 SYH-46743
            account = 15154365413
            normal_time = "2020-12-18 02:00:00%2020-12-18 10:00:00&2020-12-16 13:00:00%2020-12-16 23:00:00"
            sick_time = "2020-12-18 10:41:43%2020-12-18 10:42:37&2020-12-18 10:55:54%2020-12-18 10:56:31"
        if i == 2:
            continue
            # 王金柱 WJZ–32952
            account = 15847173073
            normal_time = "2020-12-20 04:00:00%2020-12-20 10:00:00&2020-12-20 12:00:00%2020-12-20 15:00:00&2020-12-20 18:00:00%2020-12-20 23:00:00"
            sick_time = "2020-12-20 10:06:54%2020-12-20 10:07:41&2020-12-20 16:16:20%2020-12-20 16:16:45&2020-12-20 17:15:23%2020-12-20 17:15:43"
        if i == 3:
            # 王江曼 WJM–18003
            account = 13961327467
            normal_time = "2020-12-24 12:00:00%2020-12-24 22:00:00&2020-12-25 12:00:00%2020-12-25 20:00:00"
            # sick_time4 = "2020-12-24 23:00:35%2020-12-24 23:04:03&2020-12-25 20:38:44%2020-12-25 20:40:45"
            sick_time = "2020-12-24 23:00:35%2020-12-24 23:02:35&2020-12-25 20:39:24%2020-12-25 20:40:24"
        if i == 4:
            continue
            # 董玉惠 DYH-46815
            account = 15511931800
            normal_time = "2020-12-26 01:00:00%2020-12-26 5:00:00&2020-12-26 08:00:00%2020-12-26 16:00:00"
            sick_time = "2020-12-26 06:58:37%2020-12-26 06:59:15"
        if i == 5:
            continue
            # 陆艳 LY-46529
            account = 15820761208
            normal_time = "2020-12-31 07:00:00%2020-12-31 18:00:00&2021-01-03 02:00:00%2021-01-03 04:00:00"
            sick_time = "2020-12-31 06:01:59%2020-12-31 06:02:57&2021-01-03 05:19:00%2021-01-03 05:19:46"
        if i == 6:
            continue
            # 孙爽 SS-46701
            account = 13796889768
            normal_time = "2020-12-30 01:00:00%2020-12-30 06:00:00&2020-12-30 19:00:00%2020-12-30 23:00:00"
            sick_time = "2020-12-30 06:38:04%2020-12-30 06:38:51&2020-12-30 18:03:41%2020-12-30 18:04:25"
        if i == 7:
            continue
            # 段怡欣 DYX-46850
            account = 18732141909
            normal_time = "2021-01-10 08:00:00%2021-01-10 16:00:00"
            sick_time = "2021-01-10 16:07:18%2021-01-10 16:07:35&2021-01-10 19:51:39%2021-01-10 19:52:50&2021-01-11 03:12:39%2021-01-11 03:13:42"
        if i == 8:
            continue
            # 秦硕 QS-46883
            account = 13930692558
            normal_time = "2021-01-24 02:00:00%2021-01-24 21:00:00"
            sick_time = "2021-01-24 22:59:04%2021-01-24 23:01:27&2021-01-24 21:49:41%2021-01-24 21:50:30"
        if i == 9:
            # LHR LHR-47347
            account = 17637907651
            normal_time = "2021-02-24 01:00:00%2021-02-24 08:10:00&2021-02-24 08:13:00%2021-02-24 08:30:00"
            # normal_time = "2021-02-24 06:00:00%2021-02-24 08:10:00&2021-02-24 08:13:00%2021-02-24 08:30:00" #
            sick_time = "2021-02-24 08:11:32%2021-02-24 08:12:44"
        if i == 10:
            continue
            # 范云珍 FYZ-47345
            account = 15558396067
            normal_time = "2021-03-12 01:00:00%2021-03-12 14:00:00"
            sick_time = "2021-03-12 14:59:28%2021-03-12 15:00:04"
        if i == 11:
            # 贺晓丽 HXL-48229
            account = 15805363569
            normal_time = "2021-04-22 00:50:00%2021-04-22 12:00:00"
            sick_time = "2021-04-22 00:31:46%2021-04-22 00:32:54"
        if i == 12:
            continue
            # 杜若翔 DRX-40426
            account = 18638113708
            normal_time = "2021-04-30 04:00:00%2021-04-30 18:00:00&2021-05-01 05:00:00%2021-05-01 08:00:00&2021-05-01 09:00:00%2021-05-01 10:00:00"
            sick_time = "2021-04-30 18:43:31%2021-04-30 18:44:34&2021-05-01 03:58:22%2021-05-01 04:00:27&2021-05-01 19:57:17%2021-05-01 19:59:10"
        if i == 13:
            continue
            # XYP XYP-45924
            account = 13474285782
            normal_time = "2021-05-20 00:00:00%2021-05-20 07:00:00"
            sick_time = "2021-05-20 07:31:48%2021-05-20 07:31:58&2021-05-20 10:39:27%2021-05-20 10:39:58"
        if i == 14:
            continue
            # 郭文秀 GWX-48491
            account = 15036067997
            normal_time = "2021-05-30 06:00:00%2021-05-30 22:00:00"
            sick_time = "2021-05-30 04:38:39%2021-05-30 04:40:15"
        if i == 15:
            continue
            # 季琪珂 JQK-50436
            account = 13921976212
            normal_time = "2021-10-20 00:00:00%2021-10-20 03:00:00"
            sick_time = "2021-10-20 03:29:15%2021-10-20 03:30:20"
        if i == 16:
            # 常亮 CL-50629
            account = 13889886092
            normal_time = "2021-10-23 06:00:00%2021-10-23 19:00:00"
            # normal_time = "2021-10-23 03:00:00%2021-10-23 19:00:00"
            sick_time = "2021-10-23 20:41:28%2021-10-23 20:43:01"
        if i == 17:
            # 梓涵正常集
            account = 18945059697
            # normal_time = "2022-03-13 09:36:00%2022-03-13 23:49:44"
            normal_time = "2022-03-13 14:25:00%2022-03-13 14:41:00&2022-03-13 22:00:00%2022-03-13 22:30:00&2022-03-14 17:37:00%2022-03-14 17:55:00&2022-03-14 22:30:00%2022-03-14 22:50:00"
            sick_time = "2022-03-10 09:36:00%2022-03-10 09:36:10"
        # if i == 18:
        #     # 梓涵正常集
        #     account = 18945059697
        #     normal_time = "2022-03-14 14:51:42%2022-03-14 23:06:20"
        #     sick_time = "2022-03-10 09:36:00%2022-03-10 09:36:10"

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
        ############     测试     #############

    normal_testdata_list = []  # 初始化正常情况下的数据列表
    sick_testdata_list = []  # 初始化发病情况下的数据列表

    # 田颖 TY–46316
    account = 13101097823
    normal_time = "2020-12-16 10:00:00%2020-12-16 10:01:00"
    sick_time = "2020-12-16 17:57:47%2020-12-16 18:00:00"

    normal_time_array = normal_time.split("&")
    for item in normal_time_array:
        normal_begin = item.split("%")[0]
        normal_end = item.split("%")[1]
        normal_tuple = function_mysql.conn_mysql(account, normal_begin, normal_end)
        normal_testdata_list.append(normal_tuple)

    sick_time_array = sick_time.split("&")
    for item in sick_time_array:
        sick_begin = item.split("%")[0]
        sick_end = item.split("%")[1]
        sick_tuple = function_mysql.conn_mysql(account, sick_begin, sick_end)
        sick_testdata_list.append(sick_tuple)
    print(len(normal_data_list), len(sick_data_list), len(normal_testdata_list), len(sick_testdata_list))

########################    数据处理    #################################


    all_normal_acc = [];all_testnormal_acc = []
    all_sick_acc = [];all_testsick_acc = []
    # for i in range(len(normal_data_list)):
    #     for j in range(len(normal_data_list[i])):
    #         mid_normal = medfilt(normal_data_list[i][j][7:-1], kernel_size=5)
    #         mid_normal = normal_data_list[i][j][7:-1] - mid_normal
    #         all_normal_acc = all_normal_acc + list(mid_normal)
    #
    # for i in range(len(sick_data_list)):
    #     for j in range(len(sick_data_list[i])):
    #         mid_sick = medfilt(sick_data_list[i][j][7:-1], kernel_size=5)
    #         mid_sick = sick_data_list[i][j][7:-1] - mid_sick
    #         all_sick_acc = all_sick_acc + list(mid_sick)

    Function01.buildPool(normal_data_list, 6)
    all_normal_acc = Function01.get_all_acc()
    Function01.buildPool(sick_data_list, 6)
    all_sick_acc = Function01.get_all_acc()
    print('滤波结束：', time.strftime('%Y-%m-%d %X', time.localtime(time.time())))

    for i in range(len(normal_testdata_list)):
        for j in range(len(normal_testdata_list[i])):
            mid_normal = medfilt(normal_testdata_list[i][j][7:-1], kernel_size=5)
            mid_normal = normal_testdata_list[i][j][7:-1] - mid_normal
            all_testnormal_acc = all_testnormal_acc + list(mid_normal)

    for i in range(len(sick_testdata_list)):
        for j in range(len(sick_testdata_list[i])):
            mid_sick = medfilt(sick_testdata_list[i][j][7:-1], kernel_size=5)
            mid_sick = sick_testdata_list[i][j][7:-1] - mid_sick
            all_testsick_acc = all_testsick_acc + list(mid_sick)

    all_normal_gsr = []; all_testnormal_gsr = []
    all_sick_gsr = []; all_testsick_gsr = []
    all_normal_hrt = []; all_testnormal_hrt = []
    all_sick_hrt = []; all_testsick_hrt = []
    all_normal_wrist = []; all_testnormal_wrist = []
    all_sick_wrist = []; all_testsick_wrist = []
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

    for i in range(len(normal_testdata_list)):
        for j in range(len(normal_testdata_list[i])):
            all_testnormal_gsr = all_testnormal_gsr + [normal_testdata_list[i][j][1]]
            all_testnormal_hrt = all_testnormal_hrt + [normal_testdata_list[i][j][2]]
            all_testnormal_wrist = all_testnormal_wrist + [normal_testdata_list[i][j][5]]

    for i in range(len(sick_testdata_list)):
        for j in range(len(sick_testdata_list[i])):
            all_testsick_gsr = all_testsick_gsr + [sick_testdata_list[i][j][1]]
            all_testsick_hrt = all_testsick_hrt + [sick_testdata_list[i][j][2]]
            all_testsick_wrist = all_testsick_wrist + [sick_testdata_list[i][j][5]]


####################      窗口采样    ###############################

    raw_normal_gsr = []; raw_testnormal_gsr = []
    raw_sick_gsr = []; raw_testsick_gsr = []
    raw_normal_hrt = []; raw_testnormal_hrt = []
    raw_sick_hrt = []; raw_testsick_hrt = []
    raw_normal_acc = []; raw_testnormal_acc = []
    raw_sick_acc = []; raw_testsick_acc = []
    raw_normal_wrist = []; raw_testnormal_wrist = []
    raw_sick_wrist = []; raw_testsick_wrist = []
    number_count = 0; number_testcount = 0
    raw_sj = 6
    number_of_point_in_each_sample = raw_sj*20  # 采样窗口大小
    for i in range(0, number_of_point_in_each_sample, 20):  # 使用重复采样策略
        for j in range(i, len(all_sick_acc) - number_of_point_in_each_sample, number_of_point_in_each_sample):
            sick_small_data_batch_acc = [float(x) for x in all_sick_acc[j: j + number_of_point_in_each_sample]]
            number_count += 1
            raw_sick_acc.append(sick_small_data_batch_acc)

    number_of_point_in_each_sample = raw_sj  # 采样窗口大小
    for i in range(number_of_point_in_each_sample):  # 使用重复采样策略
        for j in range(i, len(all_sick_gsr) - number_of_point_in_each_sample, number_of_point_in_each_sample):
            sick_small_data_batch_gsr = [float(x) for x in all_sick_gsr[j: j + number_of_point_in_each_sample]]
            sick_small_data_batch_hrt = [float(x) for x in all_sick_hrt[j: j + number_of_point_in_each_sample]]
            sick_small_data_batch_wrist = [float(x) for x in all_sick_wrist[j: j + number_of_point_in_each_sample]]
            raw_sick_gsr.append(sick_small_data_batch_gsr)  # 添加到整体数据集中
            raw_sick_hrt.append(sick_small_data_batch_hrt)
            raw_sick_wrist.append(sick_small_data_batch_wrist)

    number_of_point_in_each_sample = raw_sj*20
    for i in range(0, len(all_normal_acc) - number_of_point_in_each_sample, 20):
        normal_small_data_batch_acc = [float(x) for x in all_normal_acc[i: i + number_of_point_in_each_sample]]
        number_count += 1
        raw_normal_acc.append(normal_small_data_batch_acc)

    number_of_point_in_each_sample = raw_sj
    for i in range(0, len(all_normal_gsr) - number_of_point_in_each_sample, 1):
        normal_small_data_batch_gsr = [float(x) for x in all_normal_gsr[i: i + number_of_point_in_each_sample]]
        normal_small_data_batch_hrt = [float(x) for x in all_normal_hrt[i: i + number_of_point_in_each_sample]]
        normal_small_data_batch_wrist = [float(x) for x in all_normal_wrist[i: i + number_of_point_in_each_sample]]
        raw_normal_gsr.append(normal_small_data_batch_gsr)  # 添加到整体数据集中
        raw_normal_hrt.append(normal_small_data_batch_hrt)
        raw_normal_wrist.append(normal_small_data_batch_wrist)
  #################   测试   ##############################
    number_of_point_in_each_sample = raw_sj*20  # 采样窗口大小
    for i in range(0, number_of_point_in_each_sample, 20):  # 使用重复采样策略
        for j in range(i, len(all_testsick_acc) - number_of_point_in_each_sample, number_of_point_in_each_sample):
            sick_small_data_batch_acc = [float(x) for x in all_testsick_acc[j: j + number_of_point_in_each_sample]]
            number_testcount += 1
            raw_testsick_acc.append(sick_small_data_batch_acc)

    number_of_point_in_each_sample = raw_sj  # 采样窗口大小
    for i in range(number_of_point_in_each_sample):  # 使用重复采样策略
        for j in range(i, len(all_testsick_gsr) - number_of_point_in_each_sample, number_of_point_in_each_sample):
            sick_small_data_batch_gsr = [float(x) for x in all_testsick_gsr[j: j + number_of_point_in_each_sample]]
            sick_small_data_batch_hrt = [float(x) for x in all_testsick_hrt[j: j + number_of_point_in_each_sample]]
            sick_small_data_batch_wrist = [float(x) for x in all_testsick_wrist[j: j + number_of_point_in_each_sample]]
            raw_testsick_gsr.append(sick_small_data_batch_gsr)  # 添加到整体数据集中
            raw_testsick_hrt.append(sick_small_data_batch_hrt)
            raw_testsick_wrist.append(sick_small_data_batch_wrist)

    number_of_point_in_each_sample = raw_sj*20
    for i in range(0, len(all_testnormal_acc) - number_of_point_in_each_sample, number_of_point_in_each_sample):
        normal_small_data_batch_acc = [float(x) for x in all_testnormal_acc[i: i + number_of_point_in_each_sample]]
        number_testcount += 1
        raw_testnormal_acc.append(normal_small_data_batch_acc)

    number_of_point_in_each_sample = raw_sj
    for i in range(0, len(all_testnormal_gsr) - number_of_point_in_each_sample, number_of_point_in_each_sample):
        normal_small_data_batch_gsr = [float(x) for x in all_testnormal_gsr[i: i + number_of_point_in_each_sample]]
        normal_small_data_batch_hrt = [float(x) for x in all_testnormal_hrt[i: i + number_of_point_in_each_sample]]
        normal_small_data_batch_wrist = [float(x) for x in all_testnormal_wrist[i: i + number_of_point_in_each_sample]]
        raw_testnormal_gsr.append(normal_small_data_batch_gsr)  # 添加到整体数据集中
        raw_testnormal_hrt.append(normal_small_data_batch_hrt)
        raw_testnormal_wrist.append(normal_small_data_batch_wrist)
    print(len(raw_sick_acc), len(raw_sick_acc[0]))
    print(len(raw_normal_acc), len(raw_normal_acc[0]))
    print(len(raw_testsick_acc), len(raw_testsick_acc[0]))
    print(len(raw_testnormal_acc), len(raw_testnormal_acc[0]))
    """
    Module 4
    对得到的原始合成数据集进行特征提取
    主要用到的是function中的feature函数
    """
    feature_sick_data = []; feature_testsick_data = []
    feature_normal_data = []; feature_testnormal_data = []
    for i in range(0, len(raw_sick_gsr)):
        feature_data_gsr = function_data.feature_extraction_gsr(raw_sick_gsr[i])
        feature_data_hrt = function_data.feature_extraction_hrt(raw_sick_hrt[i])
        feature_data_acc = function_data.feature_extraction(raw_sick_acc[i])
        feature_data_wrist = function_data.feature_extraction_wrist(raw_sick_wrist[i])
        feature_data = feature_data_gsr + feature_data_hrt + feature_data_acc + feature_data_wrist
        feature_sick_data.append(feature_data)
    random.shuffle(feature_sick_data)
    for i in range(0, len(raw_normal_gsr)):
        feature_data_gsr = function_data.feature_extraction_gsr(raw_normal_gsr[i])
        feature_data_hrt = function_data.feature_extraction_hrt(raw_normal_hrt[i])
        feature_data_acc = function_data.feature_extraction(raw_normal_acc[i])
        feature_data_wrist = function_data.feature_extraction_wrist(raw_normal_wrist[i])
        feature_data = feature_data_gsr + feature_data_hrt + feature_data_acc + feature_data_wrist
        feature_normal_data.append(feature_data)
    random.shuffle(feature_normal_data)

    for i in range(0, len(raw_testsick_gsr)):
        feature_data_gsr = function_data.feature_extraction_gsr(raw_testsick_gsr[i])
        feature_data_hrt = function_data.feature_extraction_hrt(raw_testsick_hrt[i])
        feature_data_acc = function_data.feature_extraction(raw_testsick_acc[i])
        feature_data_wrist = function_data.feature_extraction_wrist(raw_testsick_wrist[i])
        feature_data = feature_data_gsr + feature_data_hrt + feature_data_acc + feature_data_wrist
        feature_testsick_data.append(feature_data)
    random.shuffle(feature_testsick_data)
    for i in range(0, len(raw_testnormal_gsr)):
        feature_data_gsr = function_data.feature_extraction_gsr(raw_testnormal_gsr[i])
        feature_data_hrt = function_data.feature_extraction_hrt(raw_testnormal_hrt[i])
        feature_data_acc = function_data.feature_extraction(raw_testnormal_acc[i])
        feature_data_wrist = function_data.feature_extraction_wrist(raw_testnormal_wrist[i])
        feature_data = feature_data_gsr + feature_data_hrt + feature_data_acc + feature_data_wrist
        feature_testnormal_data.append(feature_data)
    random.shuffle(feature_testnormal_data)

    """
    Module 5
    对特征集数据进行处理
    按照七三原则，切分得到训练集和测试集
    """
    # train_data, test_data = function_data.get_train_test_split_balance(feature_sick_data, feature_normal_data)
    for x in feature_sick_data:  # add label
        x.append(1)
    for x in feature_normal_data:
        x.append(0)
    for x in feature_testsick_data:  # add label
        x.append(1)
    for x in feature_testnormal_data:
        x.append(0)
##################    选用不同测试集（上是7/3分，下是选择的测试集）   ################################################################
    train_ill = feature_sick_data[:] # * 0.7
    train_normal = feature_normal_data[:] #
    test_ill = feature_sick_data[:]
    test_normal = feature_normal_data[:int(len(test_ill))]
    # train_ill = feature_sick_data
    # train_normal = feature_normal_data[:int(len(train_ill))]
    # test_ill = feature_testsick_data
    # test_normal = feature_testnormal_data[:int(len(test_ill))]
####################################################################################################
    test_ill.extend(test_normal)  # merge the ill and normal data
    train_ill.extend(train_normal)
    random.shuffle(test_ill)
    random.shuffle(train_ill)
    train_data = train_ill
    test_data = test_ill
    print(len(test_data[0]), len(train_data[0]))
    # feature_sick_data.extend(feature_normal_data[:int(len(feature_sick_data))])
    # train_data = feature_sick_data
    # feature_testsick_data.extend(feature_testnormal_data[:int(len(feature_testsick_data))])
    # test_data = feature_testsick_data
    print(len(train_data), len(test_data))
    #
    """
    Module 6
    加载相关数据并进行模型(SVC)训练、生成SVC决策边界并进行决策测试
    学长原程序中，使用了，逻辑回归、SVC、SVM线性、随机森林等模型
    对此、保留了逻辑回归、SVC、SVM线性三种，选择其中之一使用时进行注释即可，用于测试特征效度的随机森林没使用上暂时删除
    嵌入式应用时只能使用逻辑回归和SVM线性
    """

    # 加载相关数据
    feature, label = function_data.split_feature_label(train_data)
    ## print(len(label), float(len(label) - sum(label)), sum(label))
    print('The ratio between P sample and N sample in Train Set is ', sum(label) / float(len(label) - sum(label)))
    test_fea, test_la = function_data.split_feature_label(test_data)
    ## print(len(test_la), float(len(test_la) - sum(test_la)), sum(test_la))
    print('The ratio between P sample and N sample in Test Set is ', sum(test_la) / float(len(test_la) - sum(test_la)))

    #############################################   降维  ###########################
    """
    特征工程：标准化
    """
    transfer = StandardScaler()
    feature = transfer.fit_transform(feature)
    test_fea = transfer.transform(test_fea)

    feature[np.isnan(feature)] = 0
    test_fea[np.isnan(test_fea)] = 0
    ##########################    特征删除    #############################
    # kk_fea = (1, 2, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60,61,62,63,64,66,67,68,69,70,71,72,75,76,77,78,80,81,82,83,84,85,86,87,88,89,90)
    kk_fea = (26, 60, 52, 48, 49, 44, 31, 57, 50, 59, 38, 30, 35, 51, 29, 43, 28, 58, 6, 36)
    kk = [int(kk_fea[i]) - 1 for i in range(len(kk_fea))]  # 程序表示值比实际特征值少1
    kk = np.array(kk)

    feature = np.delete(feature, kk, 1)
    test_fea = np.delete(test_fea, kk, 1)
    sum_fea = len(feature[0])

    ######################   特征降维   ##########################
    feature_fuben = feature
    test_fea_fuben = test_fea
    gzlj = os.getcwd()  # 获取当前工作路径
    tzpx = []  #特征排序
    kk = (-1)  #kk是下标，k是特征数
    # 随机森林
    names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
             '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
             '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
             '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
             '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
             '83', '84', '85', '86', '87', '88', '89', '90', '91']

    for dxh in range(1):
        print('第' + str(dxh + 1) + '次')

        if dxh == 0:
            kk = np.delete(kk, 0)
        else:

            feature = feature_fuben
            test_fea = test_fea_fuben

            k = int(tzpx[-1][1])
            kk = np.append(kk, k - 1)
            kk = sorted(kk, reverse=False)
            # print('删除第' + str(kk) + '个特征')

            feature = np.delete(feature, kk, 1)
            test_fea = np.delete(test_fea, kk, 1)
            names.remove(str(k))
            kk_fea = [int(kk[i])+1 for i in range(len(kk))]    # 实际特征值比程序表示值多1
            print(str(91-len(kk)) + '个特征')
            print('特征为' + str(names))
            print('删除了' + str(kk_fea) + '特征')

        rf = RandomForestClassifier(n_estimators=301, bootstrap=True, oob_score=True, max_depth=31, max_features='sqrt')
        rf.fit(feature, label)
        # print(rf.oob_score_)

        # 特征重要性表示
        tzpx = sorted(zip(map(lambda feature: round(feature, 4), rf.feature_importances_), names), reverse=True)
        print(tzpx)
        y_test_pred = rf.predict(test_fea)
        # print(metrics.confusion_matrix(test_la, y_test_pred))
        # print(metrics.classification_report(test_la, y_test_pred))

        sum_fea = len(feature[0])
        ##########################    将发病数据与未发病数据分开    #############################
        sick_data_border = []
        normal_data_border = []
        sick_data_border_test = []
        normal_data_border_test = []
        """
        将发病数据与未发病数据分开，便于检验边界函数
        """
        for k in range(len(label)):
            if label[k] == 1:
                sick_data_border.append(feature[k])
            else:
                normal_data_border.append(feature[k])

        for l in range(len(test_la)):
            if test_la[l] == 1:
                sick_data_border.append(test_fea[l])
                sick_data_border_test.append(test_fea[l])
            else:
                normal_data_border.append(test_fea[l])
                normal_data_border_test.append(test_fea[l])
    #############################################   输出模型   ###########################
        folder_y = 'model_6条数据_全部（梓涵全+同学）_06.26'                    ######################  模型文件夹名称
        gzlj = os.getcwd()  # 获取当前工作路径

    ###################     SVM_linear    #################################################

        folder1 = folder_y + '_svmlinear'
        print('svm_linear算法：')
        print(time.strftime('%Y-%m-%d %X', time.localtime(time.time())))
        print(folder1)
        if not os.path.exists(folder1):
            os.makedirs(folder1)
        os.chdir(folder1)
        joblib.dump(transfer, 'standardScaler.model')
        svm = SVC(kernel='linear')
        c_range = np.arange(1, 12, 1)  # 指定自动调参的参数范围
        param_grid = [{'kernel': ['linear'], 'C': c_range}]
        grid = GridSearchCV(svm, param_grid, scoring='f1_macro', cv=3, n_jobs=-1)
        # 模型优劣的衡量指标为召回率，也就是正样本有多少被检测出来
        clf = grid.fit(feature, label)
        print("The bese param is", grid.best_params_)
        score = grid.score(test_fea, test_la)
        print('精度为%s' % score)

        # 使用最优参数代入手动测试，并得到决策边界参数
        svm = SVC(C=grid.best_params_['C'], kernel=grid.best_params_['kernel'])
        svm.fit(feature, label)
        y_pred = svm.predict(test_fea)
        with open('gen.txt', 'a') as f:
            print('svm线性结果', sum_fea, file=f)
            print(metrics.confusion_matrix(test_la, y_pred), file=f)
            print(metrics.classification_report(test_la, y_pred), file=f)
            f.close()
        print(metrics.confusion_matrix(test_la, y_pred))
        print(metrics.classification_report(test_la, y_pred))
        w = svm.coef_
        b = svm.intercept_
        print('the w is', w)
        print('the b is', b)

        joblib.dump(svm, 'svm.model')
        function_test.test_boundary(sick_data_border, normal_data_border, w, b, sum_fea)
        os.chdir(gzlj)  # 返回工作路径
        print()
    ###################     lagistic    #################################################
        folder2 = folder_y + '_logic'
        print('logic算法：')
        print(time.strftime('%Y-%m-%d %X', time.localtime(time.time())))
        print(folder2)
        if not os.path.exists(folder2):
            os.makedirs(folder2)
        os.chdir(folder2)
        joblib.dump(transfer, 'standardScaler.model')
        """
        训练逻辑回归模型
        """
        print('测试样本数', len(test_fea))
        print('测试1的数量', sum(test_la))
        lagistic = LogisticRegression(multi_class='ovr', C=5, solver='liblinear', max_iter=10000,
                                   class_weight='balanced')
        lagistic.fit(feature, label)
        joblib.dump(lagistic, 'lagistic.model')
        # # 预测结果
        y_pred = lagistic.predict(test_fea)

        print('The result of the logistic regression model is: ')
        print('In the predict ,%d is positive ' % sum(y_pred))
        with open('gen.txt', 'a') as f:
            print('logistic结果', file=f)
            print(metrics.confusion_matrix(test_la, y_pred), file=f)  # 计算混淆矩阵
            print(metrics.classification_report(test_la, y_pred), file=f)
            f.close()
        print(metrics.confusion_matrix(test_la, y_pred))  # 计算混淆矩阵
        print(metrics.classification_report(test_la, y_pred))

        w = lagistic.coef_  ##决策函数中的特征系数（权值）
        b = lagistic.intercept_  ##决策函数中的截距
        print('the w is', w)
        print('the b is', b)

        function_test.test_boundary(sick_data_border, normal_data_border, w, b, sum_fea)
        os.chdir(gzlj)  # 返回工作路径
        print()
    ###################     adaboost    #################################################
        folder3 = folder_y + '_adaboost'
        print('adaboost算法：')
        print(time.strftime('%Y-%m-%d %X', time.localtime(time.time())))
        print(folder3)
        if not os.path.exists(folder3):
            os.makedirs(folder3)
        os.chdir(folder3)
        joblib.dump(transfer, 'standardScaler.model')
        adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
        adaboost.fit(feature, label)
        joblib.dump(adaboost, 'adaboost.model')
        # 对测试集做预测
        y_pred = adaboost.predict(test_fea)
        # 评估预测结果
        with open('gen.txt', 'a') as f:
            print('adaboost结果', file=f)
            print(metrics.confusion_matrix(test_la, y_pred), file=f)  # 计算混淆矩阵
            print(metrics.classification_report(test_la, y_pred), file=f)
            f.close()
        print(metrics.confusion_matrix(test_la, y_pred))  # 计算混淆矩阵
        print(metrics.classification_report(test_la, y_pred))
        os.chdir(gzlj)  # 返回工作路径
        print()
    ###################     xgboost    #################################################
        folder4 = folder_y + '_xgboost'
        print('xgboost算法：')
        print(time.strftime('%Y-%m-%d %X', time.localtime(time.time())))
        print(folder4)
        if not os.path.exists(folder4):
            os.makedirs(folder4)
        os.chdir(folder4)
        joblib.dump(transfer, 'standardScaler.model')
        xgboost = XGBClassifier(use_label_encoder=False)
        xgboost.fit(feature, label)
        joblib.dump(xgboost, 'xgboost.model')
        # 对测试集做预测
        y_pred = xgboost.predict(test_fea)
        # 评估预测结果
        with open('gen.txt', 'a') as f:
            print('xgboost结果', file=f)
            print(metrics.confusion_matrix(test_la, y_pred), file=f)
            print(metrics.classification_report(test_la, y_pred), file=f)
            f.close()
        print(metrics.confusion_matrix(test_la, y_pred))  # 计算混淆矩阵
        print(metrics.classification_report(test_la, y_pred))
        os.chdir(gzlj)  # 返回工作路径
        print()
    ###################     SVM_rbf    #################################################
        # folder5 = folder_y + '_rbf'
        # print('svm_rbf算法：')
        # print(time.strftime('%Y-%m-%d %X', time.localtime(time.time())))
        # print(folder5)
        # if not os.path.exists(folder5):
        #     os.makedirs(folder5)
        # os.chdir(folder5)
        # joblib.dump(transfer, 'standardScaler.model')
        # """
        # 训练SVM线性模型
        # """
        # svc = SVC(kernel='rbf')  # 选择rbf核，也可以选择其他核
        # # 自动调参
        # c_range = np.arange(1, 12, 1)  # 指定自动调参的参数范围
        # gamma_range = np.arange(1, 1001, 100)  # 指定自动调参的参数范围
        # param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
        # grid = GridSearchCV(svc, param_grid, scoring='f1_macro', cv=3, n_jobs=-1)
        # # 模型优劣的衡量指标为召回率，也就是正样本有多少被检测出来
        # clf = grid.fit(feature, label)
        # print("The bese param is", grid.best_params_)
        # score = grid.score(test_fea, test_la)
        # print('精度为%s' % score)
        #
        # # 以下是使用最优参数代入SVC的参数并进行手动训练
        # svc = SVC(C=grid.best_params_['C'], kernel=grid.best_params_['kernel'],
        #           gamma=grid.best_params_['gamma'])
        # svc.fit(feature, label)  # 训练
        # joblib.dump(svc, 'rbf.model')
        # y_pred = svc.predict(test_fea)  # 在测试集上预测
        # with open('gen.txt', 'a') as f:
        #     print('svc核函数结果', file=f)
        #     print(metrics.confusion_matrix(test_la, y_pred), file=f)  # 计算混淆矩阵
        #     print(metrics.classification_report(test_la, y_pred), file=f)
        #     f.close()
        # print(metrics.confusion_matrix(test_la, y_pred))  # 衡量预测结果，混淆矩阵
        # print(metrics.classification_report(test_la, y_pred))
        # os.chdir(gzlj)  # 返回工作路径
        # print()

    ####################################################################
    #########################     验证    ###############################
        fea_yy_data2 = sick_data_border_test + normal_data_border_test

        svm_zq0 = 0;lag_zq0 = 0;ada_zq0 = 0;xg_zq0 = 0;svc_zq0 = 0
        svm_zq1 = 0;lag_zq1 = 0;ada_zq1 = 0;xg_zq1 = 0;svc_zq1 = 0
        for x in range(len(fea_yy_data2)):
            svm_pred = svm.predict(fea_yy_data2[x:x + 1])
            lagistic_pred = lagistic.predict(fea_yy_data2[x:x + 1])
            adaboost_pred = adaboost.predict(fea_yy_data2[x:x + 1])
            xgboost_pred = xgboost.predict(fea_yy_data2[x:x + 1])
            # svc_pred = svc.predict(fea_yy_data2[x:x + 1])

            if x < len(fea_yy_data2)/2:
                if svm_pred == 1:
                    svm_zq1 += 1
                if lagistic_pred == 1:
                    lag_zq1 += 1
                if adaboost_pred == 1:
                    ada_zq1 += 1
                if xgboost_pred == 1:
                    xg_zq1 += 1
                # if svc_pred == 1:
                #     svc_zq1 += 1
            else:
                if svm_pred == 0:
                    svm_zq0 += 1
                if lagistic_pred == 0:
                    lag_zq0 += 1
                if adaboost_pred == 0:
                    ada_zq0 += 1
                if xgboost_pred == 0:
                    xg_zq0 += 1
                # if svc_pred == 0:
                #     svc_zq0 += 1

        sum_cs_b = (len(fea_yy_data2) + 1) / 2  # 正或负样本数量，为总测试数量的一半
        svm_zql = (svm_zq0 + svm_zq1) / (sum_cs_b * 2)
        lag_zql = (lag_zq0 + lag_zq1) / (sum_cs_b * 2)
        ada_zql = (ada_zq0 + ada_zq1) / (sum_cs_b * 2)
        xg_zql = (xg_zq0 + xg_zq1) / (sum_cs_b * 2)
        # svc_zql = (svc_zq0 + svc_zq1) / (sum_cs_b * 2)

        svm_jql = svm_zq1 / (svm_zq1 + sum_cs_b - svm_zq0)
        lag_jql = lag_zq1 / (lag_zq1 + sum_cs_b - lag_zq0)
        ada_jql = ada_zq1 / (ada_zq1 + sum_cs_b - ada_zq0)
        xg_jql = xg_zq1 / (xg_zq1 + sum_cs_b - xg_zq0)
        # svc_jql = svc_zq1 / (svc_zq1 + sum_cs_b - svc_zq0)

        svm_zhl = svm_zq1 / sum_cs_b
        lag_zhl = lag_zq1 / sum_cs_b
        ada_zhl = ada_zq1 / sum_cs_b
        xg_zhl = xg_zq1 / sum_cs_b
        # svc_zhl = svc_zq1 / sum_cs_b

        print('svm准确率:', svm_zql, 'svm精确率:', svm_jql, 'svm召回率:', svm_zhl)
        # print('lag准确率:', lag_zql, 'lag精确率:', lag_jql, 'lag召回率:', lag_zhl)
        print('ada准确率:', ada_zql, 'ada精确率:', ada_jql, 'ada召回率:', ada_zhl)
        print('xg准确率:', xg_zql, 'xg精确率:', xg_jql, 'xg召回率:', xg_zhl)
        # print('svc准确率:', svc_zql, 'svc精确率:', svc_jql, 'svc召回率:', svc_zhl)

        print()
        print('3种算法，预测0和1的准确数量如下：')
        print(svm_zq0, svm_zq1)
        # print(lag_zq0, lag_zq1)
        print(ada_zq0, ada_zq1)
        print(xg_zq0, xg_zq1)
        # print(svc_zq0, svc_zq1)

