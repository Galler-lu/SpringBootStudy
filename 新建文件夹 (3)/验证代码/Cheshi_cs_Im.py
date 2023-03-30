import sys
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

# import openpyxl
import pd16_function

from scipy.signal import medfilt
from sklearn.utils.extmath import safe_sparse_dot
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
import joblib
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import deal_data_function
import Function01

if __name__ == "__main__":

    filename = '0号病人_91特征.txt'
    with open(filename, 'a') as f:
        f.close()
    with open(filename, 'w') as file_object:

        for dxh in range(30):
            print('第' + str(dxh + 1) + '次')
            # account = 17637907651
            # test_time = "2021-02-24 08:00:00%2021-02-24 09:00:00"  # 第一次发病区间

            # 患者0
            if dxh == 0:  #### 1 ##
                # continue
                account = 13101097823
                test_time = "2020-12-16 00:00:00%2020-12-16 23:59:59"  # 第一次发病区间
            # 患者1
            elif dxh == 1:
                continue
                account = 15154365413
                # test_time = "2020-12-18 10:00:00%2020-12-18 11:00:00"  # 第一次发病区间
                test_time = "2020-12-18 00:00:00%2020-12-18 14:43:33"  # 第一次发病区间  完成
            # 患者2
            elif dxh == 2:
                continue
                account = 15847173073
                # test_time = "2020-12-20 10:00:00%2020-12-20 11:00:00"  # 第一次发病区间
                test_time = "2020-12-20 00:00:00%2020-12-20 23:59:59"  # 第一次发病区间  2、3、4三合一完成
            elif dxh == 3:
                continue
                account = 15847173073
                test_time = "2020-12-20 16:00:00%2020-12-20 17:00:00"  # 第二次发病区间
            elif dxh == 4:
                continue
                account = 15847173073
                test_time = "2020-12-20 17:00:00%2020-12-20 18:00:00"  # 第三次发病区间
            # 患者3
            elif dxh == 5:
                # continue
                # continue
                account = 13961327467
                test_time = "2020-12-24 00:00:00%2020-12-24 23:59:59"  # 第一次发病区间##################好了
            elif dxh == 6:
                # continue
                # continue
                account = 13961327467
                test_time = "2020-12-25 00:00:00%2020-12-25 23:59:59"  # 第二次发病区间#############好了
            # 患者4
            elif dxh == 7:
                continue
                account = 15511931800
                test_time = "2020-12-26 00:00:00%2020-12-26 09:36:11"  # 第一次发病区间 完成
            # 患者5
            elif dxh == 8:
                continue
                account = 15820761208
                test_time = "2020-12-31 00:00:00%2020-12-31 23:59:59"  # 第一次发病区间  完成
            elif dxh == 9:
                # continue
                account = 15820761208
                test_time = "2021-01-03 00:00:00%2021-01-03 23:59:59"  # 第二次发病区间  完成
            # 患者6
            elif dxh == 10:
                continue
                account = 13796889768
                test_time = "2020-12-30 00:00:00%2020-12-30 23:59:59"  # 第一次发病区间 10、11二合一完成
            elif dxh == 11:
                continue
                account = 13796889768
                test_time = "2020-12-30 17:30:00%2020-12-30 18:30:00"  # 第二次发病区间
            # 患者7
            elif dxh == 12:
                continue
                account = 18732141909
                test_time = "2021-01-10 00:00:00%2021-01-10 23:59:59"  # 第一次发病区间 12、13二合一 完成
            elif dxh == 13:
                continue
                account = 18732141909
                test_time = "2021-01-10 19:00:00%2021-01-10 20:00:00"  # 第二次发病区间
            elif dxh == 14:
                continue
                account = 18732141909
                test_time = "2021-01-11 00:00:00%2021-01-11 23:58:22"  # 第三次发病区间 完成
            # 患者8
            elif dxh == 15:
                continue
                account = 13930692558
                test_time = "2021-01-24 00:00:00%2021-01-24 23:59:59"  # 第一次发病区间
            elif dxh == 16:
                continue
                # continue#########此处二合一
                account = 13930692558
                test_time = "2021-01-24 21:00:00%2021-01-24 22:00:00"  # 第二次发病区间
            # 患者9
            elif dxh == 17:
                # continue
                account = 17637907651
                test_time = "2021-02-24 00:00:00%2021-02-24 09:34:09"  # 第一次发病区间###########好了
            # 患者10
            elif dxh == 18:
                continue
                account = 15558396067
                test_time = "2021-03-12 09:12:08%2021-03-12 20:27:08"  # 第一次发病区间 完成
            # 患者11
            elif dxh == 19:
                # continue
                account = 15805363569
                test_time = "2021-04-22 00:00:00%2021-04-22 01:39:50"  # 第一次发病区间##############好了
            # 患者12
            elif dxh == 20:
                continue
                account = 18638113708
                test_time = "2021-04-30 00:00:00%2021-04-30 23:59:59"  # 第一次发病区间
            elif dxh == 21:
                continue
                account = 18638113708
                test_time = "2021-05-01 00:00:00%2021-05-01 23:59:59"  # 第二次发病区间 21、22二合一 完成
            elif dxh == 22:
                continue
                account = 18638113708
                test_time = "2021-05-01 19:30:00%2021-05-01 20:30:00"  # 第三次发病区间
            # 患者13
            elif dxh == 23:
                continue
                account = 13474285782
                test_time = "2021-05-20 00:00:00%2021-05-20 11:02:08"  # 第一次发病区间 23、24二合一 完成
            elif dxh == 24:
                continue
                account = 13474285782
                test_time = "2021-05-20 10:00:00%2021-05-20 11:00:00"  # 第二次发病区间
            # 患者14
            elif dxh == 25:
                # continue
                account = 15036067997
                test_time = "2021-05-30 00:00:00%2021-05-30 23:59:59"  # 第一次发病区间###########好了
            # 患者15
            elif dxh == 26:
                # continue
                account = 13921976212
                test_time = "2021-10-20 00:00:00%2021-10-20 03:37:20"  # 第一次发病区间 完成
            # 患者16
            elif dxh == 27:
                # continue
                account = 13889886092
                test_time = "2021-10-23 00:00:00%2021-10-23 23:59:59"  # 第一次发病区间############好了
            # 患者17 王彦鹏 WYP-46644
            elif dxh == 28:
                # continue
                account = 15975021597
                test_time = "2021-01-09 00:00:00%2021-01-09 23:59:59"  # 第一次发病区间##########好了
            # 患者18 简彦君 JYJ-46985
            elif dxh == 29:
                continue
                account = 14784500848
                test_time = "2021-01-20 00:00:00%2021-01-20 23:59:59"  # 第一次发病区间  完成
            elif dxh == 30:
                continue
                account = 14784500848
                test_time = "2021-01-23 00:00:00%2021-01-23 23:59:59"  ############好了
            # 患者19
            elif dxh == 31:
                account = 13838374727
                test_time = "2020-12-20 00:00:00%2020-12-20 23:59:59"  # 第一次发病区间 31、32、33、34四合一完成
            elif dxh == 32:
                account = 13838374727
                test_time = "2020-12-20 07:05:00%2020-12-20 08:05:00"  # 第二次发病区间
            elif dxh == 33:
                account = 13838374727
                test_time = "2020-12-20 11:30:00%2020-12-20 12:30:00"  # 第三、四次发病区间
            elif dxh == 34:
                account = 13838374727
                test_time = "2020-12-20 15:00:00%2020-12-20 16:00:00"  # 第五次发病区间

            test_data_list = []
            """
            下面程序主要涉及处理发病的起始时间对应的字符串，并遍历代入函数获取数据库中对应的生理数据（列表形式）
            """
            test_time_array = test_time.split("&")
            for item in test_time_array:
                test_begin = item.split("%")[0]
                test_end = item.split("%")[1]
                test_tuple = function_mysql.conn_mysql(account, test_begin, test_end)
                test_data_list.append(test_tuple)

            """
            Module 2、3
            从数据库读取的数据进行五点滤波之后直接裁切、拼接，得到处理后的normal和sick数据集
            """
            all_test_acc = []
            #
            for i in range(len(test_data_list)):
                for j in range(len(test_data_list[i])):
                    mid_test = medfilt(test_data_list[i][j][7:-1], kernel_size=5)
                    mid_test = test_data_list[i][j][7:-1] - mid_test
                    all_test_acc = all_test_acc + list(mid_test)
            ## print(all_normal[:10])

            # 测试代码（始）
            all_test_gsr = []
            all_test_hrt = []
            all_test_wrist = []
            for i in range(len(test_data_list)):
                for j in range(len(test_data_list[i])):
                    all_test_gsr = all_test_gsr + [test_data_list[i][j][1]]
                    all_test_hrt = all_test_hrt + [test_data_list[i][j][2]]
                    all_test_wrist = all_test_wrist + [test_data_list[i][j][5]]

            # 以下均为测试代码
            # 导出逻辑回归模型和标准化模型

            gzlj = os.getcwd()  # 获取当前工作路径
            shuchu_zfc_4 = ''
            for four_mx in range(3):
                # print('第' + str(four_mx + 1) + '次')
                if four_mx == 0:
                    os.chdir("model_新重叠2簇_minutes_feature36_07.19_svmlinear")
                    model = joblib.load('svm.model')
                    transfer = joblib.load('standardScaler.model')
                    impute_median = joblib.load('impute_median.model')
                    os.chdir(gzlj)

                    os.chdir("model_新重叠2簇_minutes_feature54_07.19_svmlinear")
                    model1 = joblib.load('svm.model')
                    transfer1 = joblib.load('standardScaler.model')
                    impute_median1 = joblib.load('impute_median.model')
                    os.chdir(gzlj)

                    # os.chdir("model_非重叠2簇   _19minutes_feature19_04.14_svmlinear")
                    # model2 = joblib.load('svm.model')
                    # transfer2 = joblib.load('standardScaler.model')
                    # impute_median2 = joblib.load('impute_median.model')
                    # os.chdir(gzlj)
                    #
                    # os.chdir("model_非重叠2簇   _minutes_71feature71_04.10_svmlinear")
                    # model3 = joblib.load('svm.model')
                    # transfer3 = joblib.load('standardScaler.model')
                    # impute_median3 = joblib.load('impute_median.model')
                    # os.chdir(gzlj)

                elif four_mx == 1:
                    # continue
                    os.chdir("model_新重叠2簇_minutes_feature36_07.19_logic")
                    model = joblib.load('lagistic.model')
                    transfer = joblib.load('standardScaler.model')
                    impute_median = joblib.load('impute_median.model')
                    os.chdir(gzlj)

                    os.chdir("model_新重叠2簇_minutes_feature54_07.19_logic")
                    model1 = joblib.load('lagistic.model')
                    transfer1 = joblib.load('standardScaler.model')
                    impute_median1 = joblib.load('impute_median.model')
                    os.chdir(gzlj)

                    # os.chdir("model_非重叠2簇   _19minutes_feature19_04.14_logic")
                    # model2 = joblib.load('lagistic.model')
                    # transfer2 = joblib.load('standardScaler.model')
                    # impute_median2 = joblib.load('impute_median.model')
                    # os.chdir(gzlj)
                    #
                    # os.chdir("model_非重叠2簇   _minutes_71feature71_04.10_logic")
                    # model3 = joblib.load('lagistic.model')
                    # transfer3 = joblib.load('standardScaler.model')
                    # impute_median3 = joblib.load('impute_median.model')
                    # os.chdir(gzlj)

                elif four_mx == 2:
                    # continue
                    os.chdir("model_新重叠2簇_minutes_feature36_07.19_adaboost")
                    model = joblib.load('adaboost.model')
                    transfer = joblib.load('standardScaler.model')
                    impute_median = joblib.load('impute_median.model')
                    os.chdir(gzlj)

                    os.chdir("model_新重叠2簇_minutes_feature54_07.19_adaboost")
                    model1 = joblib.load('adaboost.model')
                    transfer1 = joblib.load('standardScaler.model')
                    impute_median1 = joblib.load('impute_median.model')
                    os.chdir(gzlj)

                    # os.chdir("model_非重叠2簇   _19minutes_feature19_04.14_adaboost")
                    # model2 = joblib.load('adaboost.model')
                    # transfer2 = joblib.load('standardScaler.model')
                    # impute_median2 = joblib.load('impute_median.model')
                    # os.chdir(gzlj)
                    #
                    # os.chdir("model_非重叠2簇   _minutes_71feature71_04.10_adaboost")
                    # model3 = joblib.load('adaboost.model')
                    # transfer3 = joblib.load('standardScaler.model')
                    # impute_median3 = joblib.load('impute_median.model')
                    # os.chdir(gzlj)

                elif four_mx == 3:
                    os.chdir("model_新重叠2簇_minutes_feature36_07.19_xgboost")
                    model = joblib.load('xgboost.model')
                    transfer = joblib.load('standardScaler.model')
                    impute_median = joblib.load('impute_median.model')
                    os.chdir(gzlj)

                    os.chdir("model_新重叠2簇_minutes_feature54_07.19_xgboost")
                    model1 = joblib.load('xgboost.model')
                    transfer1 = joblib.load('standardScaler.model')
                    impute_median1 = joblib.load('impute_median.model')
                    os.chdir(gzlj)

                    # os.chdir("model_非重叠2簇   _19minutes_feature19_04.14_xgboost")
                    # model2 = joblib.load('xgboost.model')
                    # transfer2 = joblib.load('standardScaler.model')
                    # impute_median2 = joblib.load('impute_median.model')
                    # os.chdir(gzlj)
                    #
                    # os.chdir("model_非重叠2簇   _minutes_71feature71_04.10_xgboost")
                    # model3 = joblib.load('xgboost.model')
                    # transfer3 = joblib.load('standardScaler.model')
                    # impute_median3 = joblib.load('impute_median.model')
                    # os.chdir(gzlj)
                # elif four_mx == 4:
                #     os.chdir("model_J5_71t_02.26_rbf")
                #     model = joblib.load('rbf.model')
                #     transfer = joblib.load('standardScaler.model')
                #     os.chdir(gzlj)

                # 测试一整段数据
                feature_test_data_aaa = []
                # print(len(all_test_gsr))
                for i in range(len(all_test_gsr) - 6):
                    all_test_gsr_bbb = all_test_gsr[i:i + 6]
                    all_test_hrt_bbb = all_test_hrt[i:i + 6]
                    all_test_wrist_bbb = all_test_wrist[i:i + 6]
                    all_test_acc_bbb = all_test_acc[(i * 20):(i + 6) * 20]
                    feature_test_data_bbb = deal_data_function.deal_data_function(all_test_gsr_bbb, all_test_hrt_bbb,
                                                                                  all_test_acc_bbb, all_test_wrist_bbb)
                    feature_test_data_aaa.append(feature_test_data_bbb)
                feature_test_data_aaa = transfer.transform(feature_test_data_aaa)
                feature_test_data_aaa = impute_median.transform(feature_test_data_aaa)

                feature_test_data_aaa1 = feature_test_data_aaa
                # feature_test_data_aaa2 = feature_test_data_aaa
                # feature_test_data_aaa3 = feature_test_data_aaa
                # 模型一特征删除

                kk_fea = (
                    1, 2, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36,
                    37, 43, 44,
                    45,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 74, 83, 84, 85, 86, 89, 90)
                # kk_fea = (26, 60, 52, 48, 49, 44, 31, 57, 50, 59, 38, 30, 35, 51, 29, 43, 28, 58, 6, 36)
                kk = [int(kk_fea[i]) - 1 for i in range(len(kk_fea))]  # 程序表示值比实际特征值少1
                kk = np.array(kk)
                feature_test_data_aaa = np.delete(feature_test_data_aaa, kk, 1)

                # # 模型二特征删除
                kk_fea = (
                    3, 4, 8, 9, 10, 11, 12, 22, 23, 33, 38, 39, 40, 41, 42, 61, 62, 63, 66, 67, 68, 69, 70, 71, 72, 73,
                    75, 76, 77,
                    78,
                    79, 80, 81, 82, 87, 88)
                # kk_fea = (26, 60, 52, 48, 49, 44, 31, 57, 50, 59, 38, 30, 35, 51, 29, 43, 28, 58, 6, 36)
                kk = [int(kk_fea[i]) - 1 for i in range(len(kk_fea))]  # 程序表示值比实际特征值少1
                kk = np.array(kk)
                feature_test_data_aaa1 = np.delete(feature_test_data_aaa1, kk, 1)

                ##模型三特征删除
                # kk_fea = (
                # 1, 2, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                # 35, 36, 37, 39, 40, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64,
                # 65, 69, 70, 73, 74, 75, 76, 77, 78, 79, 80, 83, 84, 85, 86, 87, 89, 90)
                # # kk_fea = (26, 60, 52, 48, 49, 44, 31, 57, 50, 59, 38, 30, 35, 51, 29, 43, 28, 58, 6, 36)
                # kk = [int(kk_fea[i]) - 1 for i in range(len(kk_fea))]  # 程序表示值比实际特征值少1
                # kk = np.array(kk)
                # feature_test_data_aaa2 = np.delete(feature_test_data_aaa2, kk, 1)
                # # 模型四
                # kk_fea = (2, 3, 8, 9, 10, 11, 22, 37, 40, 41, 62, 65, 66, 67, 70, 71, 80, 81, 87)
                # # kk_fea = (26, 60, 52, 48, 49, 44, 31, 57, 50, 59, 38, 30, 35, 51, 29, 43, 28, 58, 6, 36)
                # kk = [int(kk_fea[i]) for i in range(len(kk_fea))]  # 程序表示值比实际特征值少1
                # kk = np.array(kk)
                # feature_test_data_aaa3 = np.delete(feature_test_data_aaa3, kk, 1)

                y_pred_bbb = model.predict(feature_test_data_aaa)
                print("模型一预测：", len(y_pred_bbb), sum(y_pred_bbb))

                y_pred_bbb1 = model1.predict(feature_test_data_aaa1)
                print("模型二预测：", len(y_pred_bbb1), sum(y_pred_bbb1))
                #
                # y_pred_bbb2 = model2.predict(feature_test_data_aaa2)
                # print("模型三预测：", len(y_pred_bbb2), sum(y_pred_bbb2))
                #
                # y_pred_bbb3 = model3.predict(feature_test_data_aaa3)
                # print("模型四预测：", len(y_pred_bbb3), sum(y_pred_bbb3))

                for four_ck in range(30):
                    co = 5 + four_ck  # 连续多少个1判断发病为1
                    m = 0  # 连续出现1的个数
                    k = 0  # 记录上一次判断为发病的时间10分钟以内
                    xgl = []  # 记录报警事件
                    for i in range(len(y_pred_bbb)):
                        res = y_pred_bbb[i]
                        if i - k < 600 and k != 0:
                            continue

                        if res == 0 or res1==0:
                            m = 0
                            continue
                        if res == 1 and res1==1:
                            m += 1
                            if m == co:
                                k = i
                                m = 0
                                pdi = pd16_function.pd16hs(account, test_time, i)
                                # pdi = i
                                xgl.append(pdi)

                    shuchu_zfc = str(len(xgl)) + ' ('
                    if len(xgl) != 0:
                        for i in range(len(xgl)):
                            if i < len(xgl) - 1:
                                shuchu_zfc = shuchu_zfc + str(xgl[i]) + '、'
                                # print(str(xgl[i])+'、',end='')
                            else:
                                shuchu_zfc = shuchu_zfc + str(xgl[i]) + ')'
                                # print(str(xgl[i]) + ')')
                    else:
                        shuchu_zfc = shuchu_zfc + ')'
                    if shuchu_zfc_4 == '':
                        shuchu_zfc_4 = shuchu_zfc
                    else:
                        shuchu_zfc_4 = shuchu_zfc_4 + ',' + shuchu_zfc
                    # shuchu_lb.append(shuchu_zfc)

            print(shuchu_zfc_4)

            file_object.write(shuchu_zfc_4 + '\n')
            # print(shuchu_lb)
            # shuchu_2_lb.append(shuchu_lb)

    print("程序运行结束")
