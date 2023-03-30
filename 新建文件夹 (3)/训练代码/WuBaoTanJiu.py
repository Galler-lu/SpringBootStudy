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

def DiaoYongShuJu(name):
    k = [];k1 = [];ii = 0
    with open(name, "r") as f1:
        for line in f1.readlines():
            ii += 1
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            m = float(line)
            if ii == 91:
                k.append(m)
                k1.append(k)
                ii = 0
                k = []
            else:
                k.append(m)
        f1.close()
    return k1


if __name__ == "__main__":
    print('开始时间：', time.strftime('%Y-%m-%d %X', time.localtime(time.time())))
    ######################################################################################

    y = '05试.txt'
    # y = '05(17.20_18.)试.txt'
    fea_y_data = []
    fea_y_data.append(DiaoYongShuJu(y))
    fea_y_data2 = fea_y_data[0]

    fea_n_data = []
    fea_n1_data = []    # 0判定为1的数据集
    fea_s_data = []
    for x in range(len(fea_y_data2)):
        if x <= 180:
            fea_n_data.append(fea_y_data2[x])
        if 280 <= x <= 460:
            fea_n1_data.append(fea_y_data2[x])
        # if x <= 115:
        #     fea_n_data.append(fea_y_data2[x])
        # if 284 <= x <= 399:
        #     fea_n1_data.append(fea_y_data2[x])
        if 1663 <= x <= 1796:
            fea_s_data.append(fea_y_data2[x])
        # else:
        #     fea_n_data.append(fea_y_data2[x])

    for x in fea_n_data:  # 增加标签
        x.append(0)
    for x in fea_n1_data:
        x.append(0)
    for x in fea_s_data:
        x.append(1)

    # fea_n_data.extend(fea_n1_data)
    random.shuffle(fea_n1_data)
    random.shuffle(fea_n_data)
    random.shuffle(fea_s_data)
##################    选用不同测试集（上是7/3分，下是选择的测试集）   ################################################################
    # test_ill = fea_s_data[:int(len(fea_s_data) * 0.3)]
    # test_normal = fea_n_data[:int(len(test_ill))]
    # train_ill = fea_s_data[int(len(fea_s_data) * 0.3):]
    # train_normal = fea_n_data[int(len(test_ill)):int(len(test_ill)) + int(len(train_ill))]
    test_ill = fea_s_data[:int(len(fea_s_data))]
    test_normal = fea_n_data[:int(len(test_ill))]
    train_ill = fea_s_data[:int(len(fea_s_data))]
    train_normal = fea_n_data[:int(len(train_ill))]
    print(len(test_ill), len(test_normal), len(train_ill), len(train_normal))
    # train_ill = feature_sick_data
    # train_normal = feature_normal_data[:int(len(train_ill))]
    # test_ill = feature_testsick_data
    # test_normal = feature_testnormal_data[:int(len(test_ill))]
####################################################################################################
    test_ill.extend(test_normal)  # merge the ill and normal data
    train_ill.extend(train_normal)
    train_data = train_ill
    test_data = test_ill
    print(len(train_data), len(test_data))

    feature, label = function_data.split_feature_label(train_data)
    print('The ratio between P sample and N sample in Train Set is ', sum(label) / float(len(label) - sum(label)))
    test_fea, test_la = function_data.split_feature_label(test_data)
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
    ##########################    特征删除    #############################  顺序为：皮电、心率、加速度、腕动次数
    # # kk_fea = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86)
    #      #, 27, 28, 29, 30, 57, 58, 59, 60, 87, 88, 89, 90, 91
    # kk_fea = (27, 28, 29, 30, 57, 58, 59, 60, 87, 88, 89, 90, 91)
    # kk = [int(kk_fea[i]) - 1 for i in range(len(kk_fea))]  # 程序表示值比实际特征值少1
    # kk = np.array(kk)
    #
    # feature = np.delete(feature, kk, 1)
    # test_fea = np.delete(test_fea, kk, 1)
    # sum_fea = len(feature[0])

    ######################   特征降维   ##########################
    feature_fuben = feature
    test_fea_fuben = test_fea
    gzlj = os.getcwd()  # 获取当前工作路径
    tzpx = []  # 特征排序
    kk = (-1)  # kk是下标，k是特征数
    # 随机森林
    names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
             '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
             '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
             '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
             '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
             '83', '84', '85', '86', '87', '88', '89', '90', '91']


    for dxh1 in range(1):
        print('第' + str(dxh1 + 1) + '次')

        if dxh1 == 0:
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
            kk_fea = [int(kk[i]) + 1 for i in range(len(kk))]  # 实际特征值比程序表示值多1
            sum_fea = len(feature[0])
            print(str(sum_fea) + '个特征')
            print('特征为' + str(names))
            print('删除了' + str(kk_fea) + '特征')

        rf = RandomForestClassifier(n_estimators=301, bootstrap=True, oob_score=True, max_depth=31,
                                    max_features='sqrt')
        rf.fit(feature, label)

        # 特征重要性表示
        tzpx = sorted(zip(map(lambda feature: round(feature, 4), rf.feature_importances_), names), reverse=True)
        print(tzpx)
        y_test_pred = rf.predict(test_fea)
        # print(metrics.confusion_matrix(test_la, y_test_pred))
        # print(metrics.classification_report(test_la, y_test_pred))

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
                sick_data_border_test.append(test_fea[l])
                sick_data_border.append(test_fea[l])
            else:
                normal_data_border_test.append(test_fea[l])
                normal_data_border.append(test_fea[l])
        sum_fea = len(feature[0])
    #############################################   输出模型   ###########################
        folder_y = 'model_' + str(sum_fea) + '_01.18'                     ######################  模型文件夹名称
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
        folder5 = folder_y + '_rbf'
        print('svm_rbf算法：')
        print(time.strftime('%Y-%m-%d %X', time.localtime(time.time())))
        print(folder5)
        if not os.path.exists(folder5):
            os.makedirs(folder5)
        os.chdir(folder5)
        joblib.dump(transfer, 'standardScaler.model')
        """
        训练SVM线性模型
        """
        svc = SVC(kernel='rbf')  # 选择rbf核，也可以选择其他核
        # 自动调参
        c_range = np.arange(1, 12, 1)  # 指定自动调参的参数范围
        gamma_range = np.arange(1, 1001, 100)  # 指定自动调参的参数范围
        param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
        grid = GridSearchCV(svc, param_grid, scoring='f1_macro', cv=3, n_jobs=-1)
        # 模型优劣的衡量指标为召回率，也就是正样本有多少被检测出来
        clf = grid.fit(feature, label)
        print("The bese param is", grid.best_params_)
        score = grid.score(test_fea, test_la)
        print('精度为%s' % score)

        # 以下是使用最优参数代入SVC的参数并进行手动训练
        svc = SVC(C=grid.best_params_['C'], kernel=grid.best_params_['kernel'],
                  gamma=grid.best_params_['gamma'])
        svc.fit(feature, label)  # 训练
        joblib.dump(svc, 'rbf.model')
        y_pred = svc.predict(test_fea)  # 在测试集上预测
        with open('gen.txt', 'a') as f:
            print('svc核函数结果', file=f)
            print(metrics.confusion_matrix(test_la, y_pred), file=f)  # 计算混淆矩阵
            print(metrics.classification_report(test_la, y_pred), file=f)
            f.close()
        print(metrics.confusion_matrix(test_la, y_pred))  # 衡量预测结果，混淆矩阵
        print(metrics.classification_report(test_la, y_pred))
        os.chdir(gzlj)  # 返回工作路径
        print()

    ####################################################################
    #########################     验证    ###############################
        fea_yy_data = []
        yy = '05(17.4_18.1)试.txt'            # 验证集
        fea_yy_data.append(DiaoYongShuJu(yy))
        fea_yy_data2 = fea_yy_data[0]

        fea_yy_data2 = transfer.transform(fea_yy_data2)
        fea_yy_data2[np.isnan(fea_yy_data2)] = 0
        p01 = [0 for i in range(len(fea_yy_data2))]
        print(len(fea_yy_data2))
        x1 = 1663 - 600 #+ 28794         # 发病区间
        x2 = 1796 - 600 #+ 28794
        svm_zq0 = 0;lag_zq0 = 0;ada_zq0 = 0;xg_zq0 = 0;svc_zq0 = 0
        svm_zq1 = 0;lag_zq1 = 0;ada_zq1 = 0;xg_zq1 = 0;svc_zq1 = 0
        svm_zql0 = 0;lag_zql0 = 0;ada_zql0 = 0;xg_zql0 = 0;svc_zql0 = 0
        svm_zql1 = 0;lag_zql1 = 0;ada_zql1 = 0;xg_zql1 = 0;svc_zql1 = 0
        for x in range(len(fea_yy_data2)):
            svm_pred = svm.predict(fea_yy_data2[x:x + 1])
            lagistic_pred = lagistic.predict(fea_yy_data2[x:x + 1])
            adaboost_pred = adaboost.predict(fea_yy_data2[x:x + 1])
            xgboost_pred = xgboost.predict(fea_yy_data2[x:x + 1])
            svc_pred = svc.predict(fea_yy_data2[x:x + 1])

            # if xgboost_pred == 1:
            #     print(x)

            if x1 <= x <= x2:
                p01[x] = 1

                if x == (x1 + x2 + 1)/2:
                    print(111)
                if svm_pred == p01[x]:
                    svm_zq1 += 1
                if lagistic_pred == p01[x]:
                    lag_zq1 += 1
                if adaboost_pred == p01[x]:
                    ada_zq1 += 1
                if xgboost_pred == p01[x]:
                    xg_zq1 += 1
                if svc_pred == p01[x]:
                    svc_zq1 += 1
            else:
                if svm_pred == p01[x]:
                    svm_zq0 += 1
                if lagistic_pred == p01[x]:
                    lag_zq0 += 1
                if adaboost_pred == p01[x]:
                    ada_zq0 += 1
                if xgboost_pred == p01[x]:
                    xg_zq0 += 1
                if svc_pred == p01[x]:
                    svc_zq0 += 1

            with open('YY(nn0121_)' + yy, 'a') as f:
                print(p01[x], svm_pred, lagistic_pred, adaboost_pred, xgboost_pred, svc_pred, file=f)
                f.close()

        svm_zql0 = svm_zq0 / int(len(fea_yy_data2) - sum(p01))
        lag_zql0 = lag_zq0 / int(len(fea_yy_data2) - sum(p01))
        ada_zql0 = ada_zq0 / int(len(fea_yy_data2) - sum(p01))
        xg_zql0 = xg_zq0 / int(len(fea_yy_data2) - sum(p01))
        svc_zql0 = svc_zq0 / int(len(fea_yy_data2) - sum(p01))

        print('svm_0:', svm_zq0, 'svm_1:', svm_zq1)
        print('lag_0:', lag_zq0, 'lag_1:', lag_zq1)
        print('ada_0:', ada_zq0, 'ada_1:', ada_zq1)
        print('xg_0:', xg_zq0, 'xg_1:', xg_zq1)
        print('svc_0:', svc_zq0, 'svc_1:', svc_zq1)
        print('数据量', len(fea_yy_data2), '发病量', sum(p01))

        svm_zql1 = svm_zq1 / int(sum(p01))
        lag_zql1 = lag_zq1 / int(sum(p01))
        ada_zql1 = ada_zq1 / int(sum(p01))
        xg_zql1 = xg_zq1 / int(sum(p01))
        svc_zql1 = svc_zq1 / int(sum(p01))

        print('svm_0准确率:', svm_zql0, 'svm_1准确率:', svm_zql1)
        print('lag_0准确率:', lag_zql0, 'lag_1准确率:', lag_zql1)
        print('ada_0准确率:', ada_zql0, 'ada_1准确率:', ada_zql1)
        print('xg_0准确率:', xg_zql0, 'xg_1准确率:', xg_zql1)
        print('svc_0准确率:', svc_zql0, 'svc_1准确率:', svc_zql1)

        with open('YY准确率(nn0121)', 'a') as f:
            print(yy, file=f)
            print('svm_0准确率:', svm_zql0, 'svm_1准确率:', svm_zql1, file=f)
            print('lag_0准确率:', lag_zql0, 'lag_1准确率:', lag_zql1, file=f)
            print('ada_0准确率:', ada_zql0, 'ada_1准确率:', ada_zql1, file=f)
            print('xg_0准确率:', xg_zql0, 'xg_1准确率:', xg_zql1, file=f)
            print('svc_0准确率:', svc_zql0, 'svc_1准确率:', svc_zql1, file=f)
            print(file=f)
            f.close()


