import math

import joblib
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
import itertools
import datetime
from sklearn.model_selection import cross_val_score

"定义核函数"
"np.linalg.norm(求范数)"
"范数的定义 eg向量x=[2,4,8]T(转至)的范数为:||x||=根号(2*2+4*4+8*8)=9.165"
"math.exp(1)返回e的一次方"


def rbf(gamma=1.0):
    def rbf_fun(x1, x2):
        return math.exp((np.linalg.norm(x1 - x2)) * (-1.0 * gamma))

    return rbf_fun


"x2.transpose()是对矩阵的转置"


def lin(offset=0):
    def lin_fun(x1, x2):
        return x1.dot(x2.transpose()) + offset

    return lin_fun


"pow(x1.dot(x2.transpose())+offset,power)指的是对得到x1.dot(x2.transpose())+offset的power次方"


def poly(power=2, offset=0):
    def poly_fun(x1, x2):
        return pow(x1.dot(x2.transpose()) + offset, power)

    return poly_fun


def sig(alpha=1.0, offset=0):
    def sig_fun(x1, x2):
        return math.tanh(alpha * 1.0 * x1.dot(x2.transpose()) + offset)

    return sig_fun


"根据输入X的大小构造核矩阵"


def kernel_matrix(x, kernel):
    mat = np.zeros((x.shape[0], x.shape[0]))
    for a in range(x.shape[0]):
        for b in range(x.shape[0]):
            mat[a][b] = kernel(x[a], x[b])
    return mat


".trace()得到矩阵的迹eg a=[[a11,a12],[a21,a22]] a的迹就是a11+a22的值"
"f_dot函数最后得到一个值"


def f_dot(kernel_mat1, kernel_mat2):
    return (kernel_mat1.dot(kernel_mat2.transpose())).trace()


def A(kernel_mat1, kernel_mat2):
    return (f_dot(kernel_mat1, kernel_mat2)) / (
        math.sqrt(f_dot(kernel_mat1, kernel_mat1) * f_dot(kernel_mat2, kernel_mat2)))


'''
求betas
1.形成一个y行y列的矩阵yyT,由y*yT得到
2.通过kernel_matrix,得到对X数据进行核函数的映射后的矩阵，
和X的行数列数相同，设为data
3.通过f_dot函数,将data和y相乘，返回相乘得到的矩阵f_mat,再返回矩阵的迹，记为J
4.通过A函数将f_mat和yyT相乘得到的迹，再除以根号下(f_mat*f_matT)*(yyT*yyTT)得到的矩阵的迹
5.将不同核函数在第四步得到的值相加，得到deno值
6.得到使用不同核函数情况下的betas值，
通过A函数将f_mat和yyT相乘得到的迹，再除以根号下
(f_mat*f_matT)*(yyT*yyTT)得到的矩阵的迹，最后除以deno
就的到每个核函数的betas值了
betas值是每个核函数的比重
'''


def beta_finder(x, y, kernel_list):
    y = np.matrix(y)
    yyT = y.dot(y.transpose())
    deno = sum([A(kernel_matrix(x, kernel), yyT) for kernel in kernel_list])
    betas = [A(kernel_matrix(x, kernel), yyT) / deno for kernel in kernel_list]
    print(betas)
    return betas


"产生multi核"
'''
1.得到betas
2.生成矩阵XxY维的矩阵
3.得到不同核函数对X数据映射后的数据data，再乘以该核函数对应的beta值（比重），再
4.将上述得到的矩阵相加得到最融合的矩阵
'''


def multi_kernel_maker(x, y, kernel_list):
    betas = [float(b) for b in beta_finder(x, y, kernel_list)]

    # print " ",betas
    def multi_kernal(x1, x2):
        mat = np.zeros((x1.shape[0], x2.shape[0]))
        for a in range(x1.shape[0]):
            for b in range(x2.shape[0]):
                mat[a][b] = sum([betas[i] * kernel(x1[a], x2[b]) for i, kernel in enumerate(kernel_list)])
        return mat

    return multi_kernal


"制造多核"
# kernels = [lin(),lin(2),poly(),poly(3),poly(4),rbf(),rbf(1.5),sig(),sig(1.5)]
kernels = [lin(), poly(), rbf(), sig(), rbf(10)]
kernel_numbers = 5
multi_kernels = [mult for mult in itertools.combinations(kernels,
                                                         kernel_numbers)]  # itertools.combinations迭代器eg。(combinations('ABC', 2))得到[('A', 'B'), ('A', 'C'), ('B', 'C')]
"训练模型"


def mk_train(x_train, y_train, multi_kernels):
    y = [[t] for t in y_train[:]]
    #  y=[[t] for t in y_train[:,i]]
    for k_list in multi_kernels:
        mk_train_start_time = datetime.datetime.now()
        multi_kernel = multi_kernel_maker(x_train, y, k_list)
        print(k_list, 'multi kernel maked! !')
        clf = svm.SVC(kernel=multi_kernel)
        results = cross_val_score(clf, x_train, y_train, scoring='accuracy', cv=10)
        import os
        # os.chdir("模型")
        # joblib.dump(clf, 'clf')
        print("*******************")
        print(results.mean())
        mk_train_end_time = datetime.datetime.now()
        print('mk_train_time:', (mk_train_end_time - mk_train_start_time).seconds, 'seconds')
        "导入数据"


def mkl(data, label):
    # file_path = r'C:\\Users\\DengBY\\Desktop\\liver_Lasso_features\\ALL_V.csv'
    # data = pd.read_csv(file_path)
    # a = pd.DataFrame(data)
    # X = a.values[:, 1:597]
    # y = a.values[:, 598]
    X = data
    y = label
    min_max_scaler = preprocessing.MinMaxScaler()  # 范围0-1缩放标准化
    X = min_max_scaler.fit_transform(X)
    # "基于Lasso的特征选择"
    lsvc = LassoCV().fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_lsvc = model.transform(X)
    df_X_lsvc = pd.DataFrame(X_lsvc)
    y = pd.DataFrame(y)
    print(y)
    print("bbbbbbbbbbbb*****************")
    b = df_X_lsvc
    print(b)
    objs = [b, y]
    # "features select 后的数据"
    data = pd.concat(objs, axis=1, join='outer', ignore_index=False,
                     keys=None, levels=None, names=None, verify_integrity=False)
    # print(data)
    # data = pd.merge(objs, how='outer', on=None)
    # '打乱数据，重新排序'
    data = data.sample(frac=1)
    # X = data.values[:, :14]
    # y = data.values[:, 15]
    X_train = data.values[:, :-2]
    print(X_train)

    y_train = data.values[:, -1]
    print(y_train)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=200)
    print('model training starting')
    mk_train(X_train, y_train, multi_kernels)
    print('model training finishing')
    # 保存日志
    import sys
    f_handler = open('out.log', 'w')
    sys.stdout = f_handler
