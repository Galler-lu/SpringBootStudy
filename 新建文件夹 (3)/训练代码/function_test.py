from function_data import *
import os
import random
import numpy as np
import math

from sklearn.utils.extmath import safe_sparse_dot
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def Logistic(feature, label, test_fea, test_la):
    # 训练逻辑回归模型
    model = LogisticRegression(multi_class='ovr', C=5, solver='liblinear', max_iter=10000, class_weight='balanced')
    logic = model.fit(feature, label)
    # print(model)
    print("There are %d test samples " % len(test_la))
    print('In the test samples , %d is positive(Ground truth) ' % sum(test_la))  ##拥有的正样本个数
    # # 预测结果
    y_pred = model.predict(test_fea)
    # print(y_pred)
    print('The result of the logistic regression model is: ')
    print('In the predict ,%d is positive ' % sum(y_pred))
    # #
    print('the window30 21feature logistic result is:')
    print(metrics.confusion_matrix(test_la, y_pred))  # 计算混淆矩阵
    print(metrics.classification_report(test_la, y_pred))

    w = model.coef_   ##决策函数中的特征系数（权值）
    b = model.intercept_    ##决策函数中的截距

    joblib.dump(logic, 'logic.model')
    return(w, b)

# 45-96行代码为测试边界函数相关的函数
def boundary(coef, intercept, support, gamma, test):
    test = test[:-1]
    print(test)
    k_list=[]
    for i in range(len(support)):
        k = math.exp(-gamma * np.square(np.linalg.norm(support[i]-test)))
        k_list.append(k)
    w = np.dot(coef, k_list)
    result_pred = w + intercept
    return result_pred

# def test_boundary(feature_sick_data, feature_normal_data, w, b,feature_sum):
def test_boundary(feature_sick_data, feature_normal_data, w, b, feature_sum):
    all_ill = feature_sick_data
    all_normal = feature_normal_data
    print('the ill data have:', len(all_ill))
    print('the normal data have:', len(all_normal))
    list1 = all_ill + all_normal
    print(len(list1))
    # 前一半为发病数据，后一半为正常数据
    number_count = 0
    number_count1 = 0
    number_count2 = 0

    for x in range(len(list1)):
        jueche = b[0]
        # jueche = 1   #防止报错
        # 线性SVM边界 or Logic边界
        for i in range(feature_sum):
            jueche= jueche + w[0][i]*list1[x][i]

        if jueche > 0:
            # 即为ill
            # print('the %d simple is ill' %y)
            if x < len(all_ill):
                number_count += 1
            else:
                number_count1 += 1
            
        else:
            # print('the %d simple is normal' %y)
            if x >= len(all_ill):
                number_count += 1
            else:
                number_count2 += 1
    print(number_count, number_count1, number_count2)
    score = number_count/len(list1)
    wubao = number_count1/len(list1)
    loubao = number_count2/len(list1)
    print('the score is:', float(score))
    print('the wubao is:', float(wubao))
    print('the loubao is:', float(loubao))
    print()

