import random
import time
import pandas as pd
import numpy as np
import torch
import xlwt
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # test1 = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
    # # random.shuffle(test1)
    # test2=[[1,2,3],[7,8,9]]
    # # print(test1)
    # test1=np.array(test1)
    # # test2=np.array(test2)
    # print(test1,type(test1))
    # print(test1[2:])
    # print(test2[1:])
    a = torch.zeros(1, 3)  # 创建2行3列元素全部为零(浮点型)的二维张量
    # # 结果为：tensor([[0., 0., 0.],
    # [0., 0., 0.]])
    b = torch.ones(1, 3)  # 创建2行3列元素全部为1(浮点型)的二维张量
    #     # 结果为：tensor([[1., 1., 1.],
    # [1., 1., 1.]])
    c = torch.zeros(3, 3)  # 创建3行3列元素全部为零(浮点型)的二维张量
        #     # 结果为：tensor([[0., 0., 0.],
        # [0., 0., 0.],
        # [0., 0., 0.]])
    print(torch.cat([a, b],1))