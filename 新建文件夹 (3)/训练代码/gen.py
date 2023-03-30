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

from xgboost import XGBClassifier
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
            # continue
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
    for i in range(len(normal_data_list)):
        for j in range(len(normal_data_list[i])):
            mid_normal = medfilt(normal_data_list[i][j][7:-1], kernel_size=5)
            mid_normal = normal_data_list[i][j][7:-1] - mid_normal
            all_normal_acc = all_normal_acc + list(mid_normal)

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
    print("all_normal_gsr的长度为{}".format(len(all_normal_gsr)))
    print(len(all_normal_gsr))
    # #
    for i in range(len(sick_data_list)):
        for j in range(len(sick_data_list[i])):
            all_sick_gsr = all_sick_gsr + [sick_data_list[i][j][1]]
            all_sick_hrt = all_sick_hrt + [sick_data_list[i][j][2]]
            all_sick_wrist = all_sick_wrist + [sick_data_list[i][j][5]]
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
    for i in range(raw_sj):
        for j in range(i, len(all_sick_acc) - number_of_point_in_each_sample, number_of_point_in_each_sample):
            sick_small_data_batch_acc = [float(x) for x in all_sick_acc[j: j + number_of_point_in_each_sample]]
            number_count += 1
            raw_sick_acc.append(sick_small_data_batch_acc)

    number_of_point_in_each_sample = raw_sj  # 采样窗口大小
    for i in range(number_of_point_in_each_sample):
        for j in range(i, len(all_sick_gsr) - number_of_point_in_each_sample, number_of_point_in_each_sample):
            sick_small_data_batch_gsr = [float(x) for x in all_sick_gsr[j: j + number_of_point_in_each_sample]]
            sick_small_data_batch_hrt = [float(x) for x in all_sick_hrt[j: j + number_of_point_in_each_sample]]
            sick_small_data_batch_wrist = [float(x) for x in all_sick_wrist[j:j + number_of_point_in_each_sample]]
            raw_sick_gsr.append(sick_small_data_batch_gsr)  # 添加到整体数据集中
            raw_sick_hrt.append(sick_small_data_batch_hrt)
            raw_sick_wrist.append(sick_small_data_batch_wrist)
    number_count = 0
    number_of_point_in_each_sample = raw_sj * 20
    for i in range(0, len(all_normal_acc) - number_of_point_in_each_sample + 1, number_of_point_in_each_sample):
        normal_small_data_batch_acc = [float(x) for x in all_normal_acc[i: i + number_of_point_in_each_sample]]
        number_count += 1
        raw_normal_acc.append(normal_small_data_batch_acc)
    number_of_point_in_each_sample = raw_sj
    for i in range(0, len(all_normal_gsr) - number_of_point_in_each_sample + 1, number_of_point_in_each_sample):
        normal_small_data_batch_gsr = [float(x) for x in all_normal_gsr[i: i + number_of_point_in_each_sample]]
        normal_small_data_batch_hrt = [float(x) for x in all_normal_hrt[i: i + number_of_point_in_each_sample]]
        normal_small_data_batch_wrist = [float(x) for x in all_normal_wrist[i:i + number_of_point_in_each_sample]]
        raw_normal_gsr.append(normal_small_data_batch_gsr)  # 添加到整体数据集中
        raw_normal_hrt.append(normal_small_data_batch_hrt)
        raw_normal_wrist.append(normal_small_data_batch_wrist)
    print("raw_normal_gsr的长度为：", len(raw_normal_gsr))
    print("raw_normal_hrt的长度为{}".format(len(raw_normal_hrt)))
    """
    Module 4
    对得到的原始合成数据集进行特征提取
    主要用到的是function中的feature函数
    """


    # #

    feature_sick_data = []
    feature_normal_data = []
    for i in range(0, len(raw_sick_gsr)):
        feature_data_gsr = function_data.feature_extraction_gsr(raw_sick_gsr[i])
        feature_data_hrt = function_data.feature_extraction_hrt(raw_sick_hrt[i])
        feature_data_acc = function_data.feature_extraction(raw_sick_acc[i])
        # feature_data_wrist = function_data.feature_extraction_wrist(raw_sick_wrist[i])
        feature_data = feature_data_gsr + feature_data_hrt + feature_data_acc
        feature_sick_data.append(feature_data)
    random.shuffle(feature_sick_data)
    for i in range(0, len(raw_normal_gsr)):
        feature_data_gsr = function_data.feature_extraction_gsr(raw_normal_gsr[i])
        feature_data_hrt = function_data.feature_extraction_hrt(raw_normal_hrt[i])
        feature_data_acc = function_data.feature_extraction(raw_normal_acc[i])
        # feature_data_wrist = function_data.feature_extraction_wrist(raw_normal_wrist[i])
        feature_data = feature_data_gsr + feature_data_hrt + feature_data_acc
        feature_normal_data.append(feature_data)
    print("正常样本的长度为{}".format(len(feature_normal_data)))
    random.shuffle(feature_normal_data)

    """
    Module 5
    对特征集数据进行处理
    按照七三原则，切分得到训练集和测试集
    # """
    for x in feature_sick_data:  # add label
        x.append(1)
    for x in feature_normal_data:
        x.append(0)
    ##################    选用不同测试集（上是7/3分，下是选择的测试集）   ################################################################
    test_ill = feature_sick_data[:int(len(feature_sick_data) * 0.3)]
    test_normal = feature_normal_data[:int(len(test_ill))]
    train_ill = feature_sick_data[int(len(feature_sick_data) * 0.3):]
    train_normal = feature_normal_data[int(len(test_ill)): int(len(test_ill) + len(train_ill) * 6)]
    ####################################################################################################
    test_ill.extend(test_normal)  # merge the ill and normal data
    train_ill.extend(train_normal)
    random.shuffle(test_ill)
    random.shuffle(train_ill)
    train_data = train_ill
    test_data = test_ill
    print(len(test_data[0]), len(train_data[0]))
    print(len(train_data), len(test_data))

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

    from sklearn.impute import SimpleImputer

    impute_median = SimpleImputer(missing_values=np.nan, strategy="median")
    feature = impute_median.fit_transform(feature)
    test_fea = impute_median.fit_transform(test_fea)

    from sklearn.preprocessing import MinMaxScaler

    transfer = MinMaxScaler()
    feature = transfer.fit_transform(feature)
    test_fea = transfer.transform(test_fea)
    kk_fea = (
        1, 2, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 43, 44,
        45,
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 74, 83, 84, 85, 86, 89, 90)
    kk = [int(kk_fea[i]) - 1 for i in range(len(kk_fea))]  # 程序表示值比实际特征值少1
    kk = np.array(kk)
    feature = np.delete(feature, kk, 1)
    test_fea = np.delete(test_fea, kk, 1)

    # #

    # ##########################    将发病数据与未发病数据分开    #############################
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
    sum_fea = 36
    gzlj = os.getcwd()  # 获取当前工作路径
    for mx in range(4):  # 不同模型
        if mx == 0:
            folder = folder_y + '_svmlinear'
            print('svm_linear算法：')
            print(time.strftime('%Y-%m-%d %X', time.localtime(time.time())))
            print(folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            os.chdir(folder)
            joblib.dump(transfer, 'standardScaler.model')
            joblib.dump(impute_median, "impute_median.model")
            """
            训练SVM线性模型
            """
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
            joblib.dump(impute_median, "impute_median.model")
            function_test.test_boundary(sick_data_border, normal_data_border, w, b, sum_fea)
            os.chdir(gzlj)  # 返回工作路径
            print()
        if mx == 1:
            folder = folder_y + '_logic'
            print('logic算法：')
            print(time.strftime('%Y-%m-%d %X', time.localtime(time.time())))
            print(folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            os.chdir(folder)
            joblib.dump(transfer, 'standardScaler.model')
            """
            训练逻辑回归模型
            """
            print('测试样本数', len(test_fea))
            print('测试1的数量', sum(test_la))
            model = LogisticRegression(multi_class='ovr', C=5, solver='liblinear', max_iter=10000,
                                       class_weight='balanced')
            lagistic = model.fit(feature, label)
            joblib.dump(lagistic, 'lagistic.model')
            joblib.dump(impute_median, "impute_median.model")
            # # 预测结果
            y_pred = model.predict(test_fea)

            print('The result of the logistic regression model is: ')
            print('In the predict ,%d is positive ' % sum(y_pred))
            with open('gen.txt', 'a') as f:
                print('logistic结果', file=f)
                print(metrics.confusion_matrix(test_la, y_pred), file=f)  # 计算混淆矩阵
                print(metrics.classification_report(test_la, y_pred), file=f)
                f.close()
            print(metrics.confusion_matrix(test_la, y_pred))  # 计算混淆矩阵
            print(metrics.classification_report(test_la, y_pred))

            w = model.coef_  ##决策函数中的特征系数（权值）
            b = model.intercept_  ##决策函数中的截距
            print('the w is', w)
            print('the b is', b)

            function_test.test_boundary(sick_data_border, normal_data_border, w, b, sum_fea)
            os.chdir(gzlj)  # 返回工作路径
            print()
        if mx == 2:
            folder = folder_y + '_adaboost'
            print('adaboost算法：')
            print(time.strftime('%Y-%m-%d %X', time.localtime(time.time())))
            print(folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            os.chdir(folder)
            joblib.dump(transfer, 'standardScaler.model')
            joblib.dump(impute_median, "impute_median.model")
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
        if mx == 3:
            folder = folder_y + '_xgboost'
            print('xgboost算法：')
            print(time.strftime('%Y-%m-%d %X', time.localtime(time.time())))
            print(folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            os.chdir(folder)
            joblib.dump(transfer, 'standardScaler.model')
            joblib.dump(impute_median, "impute_median.model")
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
            os.chdir(gzlj)  # 返回工作路径
            print()




