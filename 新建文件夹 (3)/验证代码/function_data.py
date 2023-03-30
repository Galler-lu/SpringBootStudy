import numpy as np
import pickle as pk
import os
import pylab as pl
import scipy.signal as signal
import random
import math
import matplotlib.pyplot as plt
import pylab as pl

from scipy.fftpack import fft
from sklearn import preprocessing
from scipy.stats import skew
from scipy import interpolate
from scipy.stats import kurtosis

np.seterr(divide='ignore', invalid='ignore')

def Approximate_Entropy(x, m, r=0.15):
    """
    近似熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    """
    # 将x转化为数组
    x = np.array(x)
    # 检查x是否为一维数据
    if x.ndim != 1:
        raise ValueError("x的维度不是一维")
    # 计算x的行数是否小于m+1
    if len(x) < m+1:
        raise ValueError("len(x)小于m+1")
    # 将x以m为窗口进行划分
    entropy = 0  # 近似熵
    for temp in range(2):
        X = []
        for i in range(len(x)-m+1-temp):
            X.append(x[i:i+m+temp])
        X = np.array(X)
        # 计算X任意一行数据与所有行数据对应索引数据的差值绝对值的最大值
        D_value = []  # 存储差值
        for i in X:
            sub = []
            for j in X:
                sub.append(max(np.abs(i-j)))
            D_value.append(sub)
        # 计算阈值
        F = r*np.std(x, ddof=1)
        # 判断D_value中的每一行中的值比阈值小的个数除以len(x)-m+1的比例
        num = np.sum(D_value<F, axis=1)/(len(x)-m+1-temp)
        # 计算num的对数平均值
        Lm = np.average(np.log(num + 1e-5))
        entropy = abs(entropy) - Lm
    return entropy


def Sample_Entropy(x, m, r=0.15):
    """
    样本熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    """
    # 将x转化为数组
    x = np.array(x)
    # 检查x是否为一维数据
    if x.ndim != 1:
        raise ValueError("x的维度不是一维")
    # 计算x的行数是否小于m+1
    if len(x) < m+1:
        raise ValueError("len(x)小于m+1")
    # 将x以m为窗口进行划分
    entropy = 0  # 近似熵
    for temp in range(2):
        X = []
        for i in range(len(x)-m+1-temp):
            X.append(x[i:i+m+temp])
        X = np.array(X)
        # 计算X任意一行数据与所有行数据对应索引数据的差值绝对值的最大值
        D_value = []  # 存储差值
        for index1, i in enumerate(X):
            sub = []
            for index2, j in enumerate(X):
                if index1 != index2:
                    sub.append(max(np.abs(i-j)))
            D_value.append(sub)
        # 计算阈值
        F = r*np.std(x, ddof=1)
        # 判断D_value中的每一行中的值比阈值小的个数除以len(x)-m+1的比例
        num = np.sum(D_value<F, axis=1)/(len(X)-m+1-temp)
        # 计算num的对数平均值
        Lm = np.average(np.log(num + 1e-5))
        entropy = abs(entropy) - Lm
    return entropy

def Fuzzy_Entropy(x, m, r=0.25, n=2):
    """
    模糊熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    n 计算模糊隶属度时的维度
    """
    # 将x转化为数组
    x = np.array(x)
    # 检查x是否为一维数据
    if x.ndim != 1:
        raise ValueError("x的维度不是一维")
    # 计算x的行数是否小于m+1
    if len(x) < m+1:
        raise ValueError("len(x)小于m+1")
    # 将x以m为窗口进行划分
    entropy = 0  # 近似熵
    for temp in range(2):
        X = []
        for i in range(len(x)-m+1-temp):
            X.append(x[i:i+m+temp])
        X = np.array(X)
        # 计算X任意一行数据与其他行数据对应索引数据的差值绝对值的最大值
        D_value = []  # 存储差值
        for index1, i in enumerate(X):
            sub = []
            for index2, j in enumerate(X):
                if index1 != index2:
                    sub.append(max(np.abs(i-j)))
            D_value.append(sub)
        # 计算模糊隶属度
        D = np.exp(-np.power(D_value, n)/r)
        # 计算所有隶属度的平均值
        Lm = np.average(D.ravel())
        entropy = abs(entropy) - Lm
    return entropy

def feature_extraction(sample_data):
    """
    特征提取函数，可在函数内部自定义各种特征提取操作
    后续可以考虑开源的包：tsfresh进行自动特征提取
    :param sample_data:输入数据，一个list，其中包含合成加速的的特征点，根据这些特征点来提取特征
    :return:返回一个特征list
    """
    # 可以首先做数据预处理，如果不的话，直接跳过
    # print('Data Preprocessing……')

    # 开始提取特征

    # 时域特征
    # 提取峰峰值
    peak_to_peak = max(sample_data) - min(sample_data)
    # 提取方差
    std = np.std(sample_data, ddof=1)
    # 提取平均值
    mean = np.mean(sample_data)
    # 提取能量
    energy = abs(mean) ** 2
    # 偏斜度
    # skewness = skew(sample_data)
    # 变异系数
    cv = std / mean
    # 振幅对数
    L = np.log10(max(sample_data) - min(sample_data) + 1)
    # 峰度
    # kurs = kurtosis(sample_data)
    # 平均能量比
    APR = np.sqrt((np.sum(np.square(sample_data))) * 1.0 / len(sample_data))
    # 峰值
    peak = max([np.abs(max(sample_data)), np.abs(min(sample_data))])
    # # 均峰比
    # PAR = np.square(peak / APR)

    # 提取四分位距
    Q1 = np.percentile(sample_data, 25)
    Q3 = np.percentile(sample_data, 75)
    iqr = Q3 - Q1
    # 过零点次数
    zeronum = 0
    for x in range(len(sample_data) - 1):
        if np.abs(sample_data[x]) > 0.05 and np.abs(sample_data[x + 1] > 0.05):
            if (np.sign(sample_data[x] * sample_data[x + 1]) < 0):
                zeronum += 1
    # 引入一\二阶差分
    difference1 = []
    difference2 = []

    for y in range(len(sample_data) - 1):
        a = sample_data[y + 1] - sample_data[y]
        difference1.append(a)
    for z in range(len(sample_data) - 2):
        b = sample_data[z + 2] - sample_data[z]
        difference2.append(b)
    # 一/二阶差分的均值
    d1mean = np.mean(difference1)
    d2mean = np.mean(difference2)
    # 一/二阶差分的四分位距
    d1Q1 = np.percentile(difference1, 25)
    d1Q3 = np.percentile(difference1, 75)
    d1iqr = d1Q3 - d1Q1

    d2Q1 = np.percentile(difference2, 25)
    d2Q3 = np.percentile(difference2, 75)
    d2iqr = d2Q3 - d2Q1
    # 一/二阶差分的标准差
    d1std = np.std(difference1, ddof=1)
    d2std = np.std(difference2, ddof=1)
    ##一/二阶差分的峰峰值
    d1peak = max(difference1) - min(difference1)
    d2peak = max(difference2) - min(difference2)
    # 一/二阶差分的绝对值均值
    d1absmean = np.mean(np.abs(difference1))
    d2absmean = np.mean(np.abs(difference2))
    # 总体变化量
    d1total_var = np.sum(np.abs(np.diff(sample_data))) / (peak_to_peak * (len(sample_data) - 1))
    d2total_var = np.sum(np.abs(np.diff(sample_data, n=2))) / (peak_to_peak * (len(sample_data) - 1))

    # 引入样本平方/立方/四次方
    sample2 = []
    # sample3 = []
    # sample4 = []
    for a in range(len(sample_data) - 1):
        b = sample_data[a] ** 2
        sample2.append(b)
        # c = sample_data[a] ** 3
        # sample3.append(c)
        # d = sample_data[a] ** 4
        # sample4.append(d)

    # 平方/立方的均值
    x2mean = np.mean(sample2)
    # x3mean = np.mean(sample3)
    # x4mean = np.mean(sample4)
    # 平方/立方的四分位距
    x2Q1 = np.percentile(sample2, 25)
    x2Q3 = np.percentile(sample2, 75)
    x2iqr = x2Q3 - x2Q1

    # x3Q1 = np.percentile(sample3, 25)
    # x3Q3 = np.percentile(sample3, 75)
    # x3iqr = x3Q3 - x3Q1
    #
    # x4Q1 = np.percentile(sample4, 25)
    # x4Q3 = np.percentile(sample4, 75)
    # x4iqr = x4Q3 - x4Q1
    # 平方/立方的标准差
    x2std = np.std(sample2, ddof=1)
    # x3std = np.std(sample3, ddof=1)
    # x4std = np.std(sample4, ddof=1)
    # 平方/立方的峰峰值
    x2peak = max(sample2) - min(sample2)
    # x3peak = max(sample3) - min(sample3)
    # x4peak = max(sample4) - min(sample4)

    # 频率特征
    Fs = 20  # 采样频率
    number = Fs * 6  # 采样点数
    fdata = fft(sample_data) * 2 / number  # 做快速傅里叶变化
    # freal = fdata.real  # 获取实数部分
    # fimag = fdata.imag  # 获取虚数部分

    absFdata = np.abs(fdata)  # 含有虚数的振幅值
    # angleFdata = np.angle(fdata)  # 相位
    absFdata = np.array(absFdata)  # 求振幅的绝对值

    x = np.arange(0, Fs, Fs / number)  # 频率个数
    half_x = x[range(int(number / 2))]  # 取一半区间
    normalization_half_y = absFdata[range(int(number / 2))]  # 由于对称性，只取一半区间（单边频谱）

    # 由于fft会在低频出现冲击干扰，所以需要一个高通滤波
    index1 = np.argwhere(half_x == 1)  # 寻找频率值为1的索引，此时为一个（1，1）阶的元组
    index = index1[0][0]  # 将元组中的元素提取出，此时为整型的索引
    xEnd = half_x[index:]  # 滤波后的频率值
    yEnd = normalization_half_y[index:]  # 滤波后的频域幅值

    # 峰值频点
    yfp = list(yEnd)
    fpeak = max(yfp)
    downnum = yfp.index(fpeak)
    xpeak = xEnd[downnum]

    # 功率谱密度
    psd = []
    for i in yEnd:
        i = i ** 2 / number
        psd.append(i)

    # 求功率谱密度最大值
    psd_max = max(psd)

    # 求能量谱能量值
    fenergy = 0
    for i in yEnd:
        fenergy += i ** 2

    # # 下面是非线性特征
    # # 近似熵
    # apEn = Approximate_Entropy(sample_data, 2)
    # # 样本熵
    # sampEn = Sample_Entropy(sample_data, 2)
    # # 模糊熵
    # fuzzyEn = Fuzzy_Entropy(sample_data, 2)

    # return [peak_to_peak, std, mean, energy, iqr,
    #         d1iqr, d2iqr, d1std, d2std, d1peak, d2peak, d1absmean, d2absmean, peak.real, apEn, sampEn, fuzzyEn]

    # return [peak_to_peak, std, mean, energy, iqr, zeronum, skewness, cv, L, kurs, PAR,
    #         d1mean, d2mean, d1iqr, d2iqr, d1std, d2std, d1peak, d2peak, d1absmean, d2absmean, d1total_var, d2total_var,
    #         x2mean, x3mean, x4mean, x2iqr, x3iqr, x4iqr, x2std, x3std, x4std, x2peak, x3peak, x4peak,
    #         fpeak, xpeak, psd_max, fenergy]
    return [peak_to_peak, std, mean, energy, iqr, zeronum, cv, L, APR, peak,
            d1mean, d2mean, d1iqr, d2iqr, d1std, d2std, d1peak, d2peak, d1absmean, d2absmean, d1total_var, d2total_var,
            x2mean, x2peak, x2std, x2iqr, fpeak, xpeak, psd_max, fenergy]


def get_train_test_split_balance(ill_list, normal_list, test_ritio=0.3):
    # 训练阶段正负样本比例保持一致，避免训练的结果较差
    # 但是测试阶段，负样本可以多一些，正好测试误识别率
    # random.shuffle(ill_list)
    # random.shuffle(normal_list)
    # 固定list

    for x in ill_list:  # add label
        x.append(1)
    for x in normal_list:
        x.append(0)


    test_ill = ill_list[:int(len(ill_list) * test_ritio)]
    test_normal = normal_list[:int(len(test_ill))]

    train_ill = ill_list[int(len(ill_list) * test_ritio):]
    train_normal = normal_list[int(len(test_ill)): int(len(test_ill) + len(train_ill))]
    # train_normal = normal_list[int(len(test_ill) * 1.2): int(len(test_ill) * 1.2 + len(train_ill) * 1.2)]
    # train_normal = normal_list[int(len(normal_list) * test_ritio):]
    print('测试1', len(test_ill))
    print('测试0', len(test_normal))
    print('训练1', len(train_ill))
    print('训练0', len(train_normal))

    test_ill.extend(test_normal)  # merge the ill and normal data
    train_ill.extend(train_normal)
    random.shuffle(test_ill)
    random.shuffle(train_ill)

    # save_pkl('train_data_balance.pkl', train_ill)
    # save_pkl('test_data_balance.pkl', test_ill) 
    return train_ill, test_ill  # 返回的分别是训练和测试数据 


def split_feature_label(data):
    feature = []
    label = []
    for x in data:
        feature.append(x[:-1])
        label.append(x[-1])
    return np.array(feature), np.array(label)

def interpolation(data, n):
    num = len(data)
    x = np.linspace(0, num-1, num)
    xnew = np.linspace(0, num-1, n*num)
    f = interpolate.interp1d(x, data, kind="cubic")
    ynew = f(xnew)
    return ynew

def feature_extraction_gsr(sample_data):
    """
    特征提取函数，可在函数内部自定义各种特征提取操作
    后续可以考虑开源的包：tsfresh进行自动特征提取
    :param sample_data:输入数据，一个list，其中包含合成加速的的特征点，根据这些特征点来提取特征
    :return:返回一个特征list
    """
    # 可以首先做数据预处理，如果不的话，直接跳过
    # print('Data Preprocessing……')

    # 开始提取特征

    # 时域特征
    # 提取峰峰值
    peak_to_peak = max(sample_data) - min(sample_data)
    # 提取方差
    std = np.std(sample_data, ddof=1)
    # 提取平均值
    mean = np.mean(sample_data)
    # 提取能量
    energy = abs(mean) ** 2
    # 偏斜度
    # skewness = skew(sample_data)
    # 变异系数
    cv = std / mean
    # 振幅对数
    L = np.log10(max(sample_data) - min(sample_data) + 1)
    # 峰度
    # kurs = kurtosis(sample_data)
    # 平均能量比
    APR = np.sqrt((np.sum(np.square(sample_data))) * 1.0 / len(sample_data))
    # 峰值
    peak = max([np.abs(max(sample_data)), np.abs(min(sample_data))])
    # # 均峰比
    # PAR = np.square(peak / APR)

    # 提取四分位距
    Q1 = np.percentile(sample_data, 25)
    Q3 = np.percentile(sample_data, 75)
    iqr = Q3 - Q1
    # 过零点次数
    zeronum = 0
    for x in range(len(sample_data) - 1):
        if np.abs(sample_data[x]) > 0.05 and np.abs(sample_data[x + 1] > 0.05):
            if (np.sign(sample_data[x] * sample_data[x + 1]) < 0):
                zeronum += 1
    # 引入一\二阶差分
    difference1 = []
    difference2 = []

    for y in range(len(sample_data) - 1):
        a = sample_data[y + 1] - sample_data[y]
        difference1.append(a)
    for z in range(len(sample_data) - 2):
        b = sample_data[z + 2] - sample_data[z]
        difference2.append(b)
    # 一/二阶差分的均值
    d1mean = np.mean(difference1)
    d2mean = np.mean(difference2)
    # 一/二阶差分的四分位距
    d1Q1 = np.percentile(difference1, 25)
    d1Q3 = np.percentile(difference1, 75)
    d1iqr = d1Q3 - d1Q1

    d2Q1 = np.percentile(difference2, 25)
    d2Q3 = np.percentile(difference2, 75)
    d2iqr = d2Q3 - d2Q1
    # 一/二阶差分的标准差
    d1std = np.std(difference1, ddof=1)
    d2std = np.std(difference2, ddof=1)
    ##一/二阶差分的峰峰值
    d1peak = max(difference1) - min(difference1)
    d2peak = max(difference2) - min(difference2)
    # 一/二阶差分的绝对值均值
    d1absmean = np.mean(np.abs(difference1))
    d2absmean = np.mean(np.abs(difference2))
    # 总体变化量
    d1total_var = np.sum(np.abs(np.diff(sample_data))) / (peak_to_peak * (len(sample_data) - 1))
    d2total_var = np.sum(np.abs(np.diff(sample_data, n=2))) / (peak_to_peak * (len(sample_data) - 1))

    # 引入样本平方/立方/四次方
    sample2 = []
    # sample3 = []
    # sample4 = []
    for a in range(len(sample_data) - 1):
        b = sample_data[a] ** 2
        sample2.append(b)
        # c = sample_data[a] ** 3
        # sample3.append(c)
        # d = sample_data[a] ** 4
        # sample4.append(d)

    # 平方/立方的均值
    x2mean = np.mean(sample2)
    # x3mean = np.mean(sample3)
    # x4mean = np.mean(sample4)
    # 平方/立方的四分位距
    x2Q1 = np.percentile(sample2, 25)
    x2Q3 = np.percentile(sample2, 75)
    x2iqr = x2Q3 - x2Q1

    # x3Q1 = np.percentile(sample3, 25)
    # x3Q3 = np.percentile(sample3, 75)
    # x3iqr = x3Q3 - x3Q1
    #
    # x4Q1 = np.percentile(sample4, 25)
    # x4Q3 = np.percentile(sample4, 75)
    # x4iqr = x4Q3 - x4Q1
    # 平方/立方的标准差
    x2std = np.std(sample2, ddof=1)
    # x3std = np.std(sample3, ddof=1)
    # x4std = np.std(sample4, ddof=1)
    # 平方/立方的峰峰值
    x2peak = max(sample2) - min(sample2)
    # x3peak = max(sample3) - min(sample3)
    # x4peak = max(sample4) - min(sample4)

    # 频率特征
    Fs = 1  # 采样频率
    number = Fs * 6  # 采样点数
    fdata = fft(sample_data) * 2 / number  # 做快速傅里叶变化
    # freal = fdata.real  # 获取实数部分
    # fimag = fdata.imag  # 获取虚数部分

    absFdata = np.abs(fdata)  # 含有虚数的振幅值
    # angleFdata = np.angle(fdata)  # 相位
    absFdata = np.array(absFdata)  # 求振幅的绝对值

    x = np.arange(0, Fs, Fs / number)  # 频率个数

    half_x = x[range(int(number / 2))]  # 取一半区间
    normalization_half_y = absFdata[range(int(number / 2))]  # 由于对称性，只取一半区间（单边频谱）
    # print(x,';',absFdata)
    # plt.figure()
    # plt.plot(x, absFdata)
    # plt.show()
    # 由于fft会在低频出现冲击干扰，所以需要一个高通滤波
    index1 = np.argwhere(half_x == 1/6)  # 寻找频率值为1的索引，此时为一个（1，1）阶的元组
    index = index1[0][0]  # 将元组中的元素提取出，此时为整型的索引
    xEnd = half_x[index:]  # 滤波后的频率值
    yEnd = normalization_half_y[index:]  # 滤波后的频域幅值
    # plt.figure()
    # plt.plot(xEnd, yEnd)
    # plt.show()
    # 峰值频点
    yfp = list(yEnd)
    fpeak = max(yfp)
    downnum = yfp.index(fpeak)
    xpeak = xEnd[downnum]

    # 功率谱密度
    psd = []
    for i in yEnd:
        i = i ** 2 / number
        psd.append(i)

    # 求功率谱密度最大值
    psd_max = max(psd)

    # 求能量谱能量值
    fenergy = 0
    for i in yEnd:
        fenergy += i ** 2

    # # 下面是非线性特征
    # # 近似熵
    # apEn = Approximate_Entropy(sample_data, 2)
    # # 样本熵
    # sampEn = Sample_Entropy(sample_data, 2)
    # # 模糊熵
    # fuzzyEn = Fuzzy_Entropy(sample_data, 2)

    # return [peak_to_peak, std, mean, energy, iqr,
    #         d1iqr, d2iqr, d1std, d2std, d1peak, d2peak, d1absmean, d2absmean, peak.real, apEn, sampEn, fuzzyEn]

    # return [peak_to_peak, std, mean, energy, iqr, zeronum, skewness, cv, L, kurs, PAR,
    #         d1mean, d2mean, d1iqr, d2iqr, d1std, d2std, d1peak, d2peak, d1absmean, d2absmean, d1total_var, d2total_var,
    #         x2mean, x3mean, x4mean, x2iqr, x3iqr, x4iqr, x2std, x3std, x4std, x2peak, x3peak, x4peak,
    #         fpeak, xpeak, psd_max, fenergy]
    return [peak_to_peak, std, mean, energy, iqr,zeronum, cv, L, APR, peak,
            d1mean, d2mean, d1iqr, d2iqr, d1std, d2std, d1peak, d2peak, d1absmean, d2absmean,
            d1total_var, d2total_var, x2mean, x2peak, x2std, x2iqr, fpeak, xpeak, psd_max, fenergy]
    # return [peak_to_peak, std, mean, energy, iqr, zeronum, cv, L, APR, peak,
    #             d1mean, d2mean, d1iqr, d2iqr, d1std, d2std, d1peak, d2peak, d1absmean, d2absmean, d1total_var, d2total_var,
    #             x2mean, x2peak, x2std, x2iqr]
    # return [mean, energy, iqr, cv, APR, peak, d2iqr, d2peak, d2absmean, d1total_var, x2mean, x2iqr, x2iqr]

def feature_extraction_hrt(sample_data):
    """
    特征提取函数，可在函数内部自定义各种特征提取操作
    后续可以考虑开源的包：tsfresh进行自动特征提取
    :param sample_data:输入数据，一个list，其中包含合成加速的的特征点，根据这些特征点来提取特征
    :return:返回一个特征list
    """
    # 开始提取特征

    # 时域特征
    # 提取峰峰值
    peak_to_peak = max(sample_data) - min(sample_data)
    # 提取方差
    std = np.std(sample_data, ddof=1)
    # 提取平均值
    mean = np.mean(sample_data)
    # 提取能量
    energy = abs(mean) ** 2
    # 偏斜度
    # skewness = skew(sample_data)
    # 变异系数
    cv = std / mean
    # 振幅对数
    L = np.log10(max(sample_data) - min(sample_data) + 1)
    # 峰度
    # kurs = kurtosis(sample_data)
    # 平均能量比
    APR = np.sqrt((np.sum(np.square(sample_data))) * 1.0 / len(sample_data))
    # 峰值
    peak = max([np.abs(max(sample_data)), np.abs(min(sample_data))])
    # # 均峰比
    # PAR = np.square(peak / APR)

    # 提取四分位距
    Q1 = np.percentile(sample_data, 25)
    Q3 = np.percentile(sample_data, 75)
    iqr = Q3 - Q1
    # 过零点次数
    zeronum = 0
    for x in range(len(sample_data) - 1):
        if np.abs(sample_data[x]) > 0.05 and np.abs(sample_data[x + 1] > 0.05):
            if (np.sign(sample_data[x] * sample_data[x + 1]) < 0):
                zeronum += 1
    # 引入一\二阶差分
    difference1 = []
    difference2 = []

    for y in range(len(sample_data) - 1):
        a = sample_data[y + 1] - sample_data[y]
        difference1.append(a)
    for z in range(len(sample_data) - 2):
        b = sample_data[z + 2] - sample_data[z]
        difference2.append(b)
    # 一/二阶差分的均值
    d1mean = np.mean(difference1)
    d2mean = np.mean(difference2)
    # 一/二阶差分的四分位距
    d1Q1 = np.percentile(difference1, 25)
    d1Q3 = np.percentile(difference1, 75)
    d1iqr = d1Q3 - d1Q1

    d2Q1 = np.percentile(difference2, 25)
    d2Q3 = np.percentile(difference2, 75)
    d2iqr = d2Q3 - d2Q1
    # 一/二阶差分的标准差
    d1std = np.std(difference1, ddof=1)
    d2std = np.std(difference2, ddof=1)
    ##一/二阶差分的峰峰值
    d1peak = max(difference1) - min(difference1)
    d2peak = max(difference2) - min(difference2)
    # 一/二阶差分的绝对值均值
    d1absmean = np.mean(np.abs(difference1))
    d2absmean = np.mean(np.abs(difference2))
    # 总体变化量
    d1total_var = np.sum(np.abs(np.diff(sample_data))) / (peak_to_peak * (len(sample_data) - 1))
    d2total_var = np.sum(np.abs(np.diff(sample_data, n=2))) / (peak_to_peak * (len(sample_data) - 1))

    # 引入样本平方/立方/四次方
    sample2 = []
    # sample3 = []
    # sample4 = []
    for a in range(len(sample_data) - 1):
        b = sample_data[a] ** 2
        sample2.append(b)
        # c = sample_data[a] ** 3
        # sample3.append(c)
        # d = sample_data[a] ** 4
        # sample4.append(d)

    # 平方/立方的均值
    x2mean = np.mean(sample2)
    # x3mean = np.mean(sample3)
    # x4mean = np.mean(sample4)
    # 平方/立方的四分位距
    x2Q1 = np.percentile(sample2, 25)
    x2Q3 = np.percentile(sample2, 75)
    x2iqr = x2Q3 - x2Q1

    # x3Q1 = np.percentile(sample3, 25)
    # x3Q3 = np.percentile(sample3, 75)
    # x3iqr = x3Q3 - x3Q1
    #
    # x4Q1 = np.percentile(sample4, 25)
    # x4Q3 = np.percentile(sample4, 75)
    # x4iqr = x4Q3 - x4Q1
    # 平方/立方的标准差
    x2std = np.std(sample2, ddof=1)
    # x3std = np.std(sample3, ddof=1)
    # x4std = np.std(sample4, ddof=1)
    # 平方/立方的峰峰值
    x2peak = max(sample2) - min(sample2)
    # x3peak = max(sample3) - min(sample3)
    # x4peak = max(sample4) - min(sample4)

    # 频率特征
    Fs = 1  # 采样频率
    number = Fs * 6  # 采样点数
    fdata = fft(sample_data) * 2 / number  # 做快速傅里叶变化,##xgl_归一化
    # freal = fdata.real  # 获取实数部分
    # fimag = fdata.imag  # 获取虚数部分

    absFdata = np.abs(fdata)  # 含有虚数的振幅值
    # angleFdata = np.angle(fdata)  # 相位
    absFdata = np.array(absFdata)  # 求振幅的绝对值

    x = np.arange(0, Fs, Fs / number)  # 频率个数
    half_x = x[range(int(number / 2))]  # 取一半区间
    normalization_half_y = absFdata[range(int(number / 2))]  # 由于对称性，只取一半区间（单边频谱）

    # 由于fft会在低频出现冲击干扰，所以需要一个高通滤波
    index1 = np.argwhere(half_x == 1/6)  # 寻找频率值为1的索引，此时为一个（1，1）阶的元组
    index = index1[0][0]  # 将元组中的元素提取出，此时为整型的索引
    xEnd = half_x[index:]  # 滤波后的频率值
    yEnd = normalization_half_y[index:]  # 滤波后的频域幅值

    # 峰值频点
    yfp = list(yEnd)
    fpeak = max(yfp)
    downnum = yfp.index(fpeak)
    xpeak = xEnd[downnum]

    # 功率谱密度
    psd = []
    for i in yEnd:
        i = i ** 2 / number
        psd.append(i)

    # 求功率谱密度最大值
    psd_max = max(psd)

    # 求能量谱能量值
    fenergy = 0
    for i in yEnd:
        fenergy += i ** 2

    # # 下面是非线性特征
    # # 近似熵
    # apEn = Approximate_Entropy(sample_data, 2)
    # # 样本熵
    # sampEn = Sample_Entropy(sample_data, 2)
    # # 模糊熵
    # fuzzyEn = Fuzzy_Entropy(sample_data, 2)

    # return [peak_to_peak, std, mean, energy, iqr,
    #         d1iqr, d2iqr, d1std, d2std, d1peak, d2peak, d1absmean, d2absmean, peak.real, apEn, sampEn, fuzzyEn]

    # return [peak_to_peak, std, mean, energy, iqr, zeronum, skewness, cv, L, kurs, PAR,
    #         d1mean, d2mean, d1iqr, d2iqr, d1std, d2std, d1peak, d2peak, d1absmean, d2absmean, d1total_var, d2total_var,
    #         x2mean, x3mean, x4mean, x2iqr, x3iqr, x4iqr, x2std, x3std, x4std, x2peak, x3peak, x4peak,
    #         fpeak, xpeak, psd_max, fenergy]
    return [peak_to_peak, std, mean, energy, iqr,zeronum, cv, L, APR, peak,
            d1mean, d2mean, d1iqr, d2iqr, d1std, d2std, d1peak, d2peak, d1absmean, d2absmean, d1total_var, d2total_var,
            x2mean, x2peak, x2std, x2iqr, fpeak, xpeak, psd_max, fenergy]
    # return [energy, L, d1mean, d2mean, d2std, d1peak, d2absmean, d1total_var, d2total_var, x2iqr, fpeak, psd_max,
    #         fenergy]



def feature_extraction_wrist(sample_data):
    # 频率特征
    # 做fft频谱分析
    Fs = 1  # 采样频率
    number = Fs * 6  # 采样点数
    fdata = fft(sample_data) * 2 / number  # 做快速傅里叶变化
    freal = fdata.real  # 获取实数部分
    fimag = fdata.imag  # 获取虚数部分

    absFdata = np.abs(fdata)  # 含有虚数的振幅值
    angleFdata = np.angle(fdata)  # 相位
    absFdata = np.array(absFdata)  # 求振幅的绝对值

##############   修改的   ###################
    # x = np.arange(0, Fs, Fs / number)  # 频率个数
    # # half_x = x[range(int(number / 2))]  # 取一半区间
    # normalization_half_y = absFdata[range(number)]  # 由于对称性，只取一半区间（单边频谱）
    # yfp = 20 * np.log10(np.clip(np.abs(normalization_half_y), 1e-10, 1e10))
    # # 由于fft会在低频出现冲击干扰，所以需要一个高通滤波
    # index1 = np.argwhere(x == 1/6)  # 寻找频率值为0.2的索引，此时为一个（1，1）阶的元组
    # index = index1[0][0]  # 将元组中的元素提取出，此时为整型的索引
    # xEnd = x[index:]  # 滤波后的频率值
    # yEnd = yfp[index:]  # 滤波后的频域幅值

##############   原先的   ###################
    x = np.arange(0, Fs, Fs / number)  # 频率个数
    half_x = x[range(int(number / 2))]  # 取一半区间
    normalization_half_y = absFdata[range(int(number / 2))]  # 由于对称性，只取一半区间（单边频谱）
    yfp = 20 * np.log10(np.clip(np.abs(normalization_half_y), 1e-10, 1e10))
    # 由于fft会在低频出现冲击干扰，所以需要一个高通滤波
    index1 = np.argwhere(half_x == 1/6)  # 寻找频率值为0.2的索引，此时为一个（1，1）阶的元组
    index = index1[0][0]  # 将元组中的元素提取出，此时为整型的索引
    xEnd = half_x[index:]  # 滤波后的频率值
    yEnd = yfp[index:]  # 滤波后的频域幅值



    # 峰值频点
    yEnd = list(yEnd)
    peak = max(yEnd)

    return [peak]