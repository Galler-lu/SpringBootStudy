import function_data
from scipy.signal import medfilt
import random


def deal_data_function(all_gsr_aaa, all_hrt_aaa, all_acc_aaa, all_wrist_aaa):
    """
    Module 4
    对得到的原始合成数据集进行特征提取
    主要用到的是function中的feature函数
    """
    feature_data_gsr = function_data.feature_extraction_gsr(all_gsr_aaa)
    feature_data_hrt = function_data.feature_extraction_hrt(all_hrt_aaa)
    feature_data_acc = function_data.feature_extraction(all_acc_aaa)
    # feature_data_wrist = function_data.feature_extraction_wrist(all_wrist_aaa)
    # feature_data = feature_data_gsr + feature_data_hrt + feature_data_acc + feature_data_wrist
    # feature_data = feature_data_gsr + feature_data_hrt
    # feature_data = feature_data_hrt + feature_data_gsr+feature_data_acc+feature_data_wrist
    # feature_data = feature_data_hrt + feature_data_gsr + feature_data_acc
    feature_data = feature_data_gsr + feature_data_hrt + feature_data_acc

    return feature_data
