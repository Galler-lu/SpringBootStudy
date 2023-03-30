#coding=utf-8
from multiprocessing import Pool
from scipy.signal import medfilt

"""
    往特征txt文件中写入数据，此处以发病数据为例
    # for i in range(0,len(feature_sick_data)):
    #     for j in range(0,len(feature_sick_data[i])):
    #         with open("all_sick_acc_medfit.txt","a+") as f1:
    #             f1.writelines(str(feature_sick_data[i][j])+"\n")
"""
def readData(src,numFeature):
    """

    :param src: 文件读取目录即提取到的特征txt文件目录
    :param numFeature: 所提取的特征种类数，比如加速度提取30种特征
    :return:
    """
    data = []
    feature_data = []
    flag=0
    with open(src, "r+") as f1:
        for line in f1.readlines():

            line = line.strip("\n")#除去换行符
            line = float(line)
            data.append(line)
            flag += 1
            if flag == numFeature:#满足特征数则为一个列表
                feature_data.append(data)
                flag = 0
                data = []
    return feature_data

all_acc = []
def dataProcess(data_list):
    """
    目标函数
    :param data_list:传入的应该是一个元组
    :return: 每个进程截取完加速度后的列表
    """
    mid_all_acc=[]
    for i in range(len(data_list)):
        #对一个时间段元组中的每一个时刻的元组进行截取
            data_acceleration = data_list[i][7:-1]
            data_baseline = medfilt(data_acceleration, 5)
            mid_normal = data_acceleration - data_baseline
            mid_all_acc = mid_all_acc + list(mid_normal)
    return mid_all_acc

def callbackData(dataList):
    """
    回调函数，每个进程执行完目标函数后会执行此回调函数
    :param dataList: 目标函数的返回值
    :return:
    """
    global all_acc
    all_acc=all_acc+dataList

def buildPool(data_list,num):
    """

    :param data_list: 传入要处理复杂列表
    :param num: 进程数，此值根据电脑的cpu物理内核的不同来选择，比如8内核16逻辑处理器的电脑，进程数最大不要超过8，还要结合cpu的利用率来看，此处选择的值为6
    :return:
    """
    global all_acc
    all_acc = []
    pool=Pool(num)#创建人为指定大小的线程池
    for data_list_acc in data_list:
        """
        将传入的列表的每个时间段即一个元组作为参数传入目标函数中并采用非堵塞式执行，何为非堵塞式自行百度
        """
        pool.apply_async(dataProcess,args=(data_list_acc,),callback=callbackData)
    pool.close()
    pool.join()


def get_all_acc():
    """
    获取处理后的正常或者发病加速度
    :return:
    """
    return all_acc

def up_sampling_sick(all_sick_acc,length):
    with open("length.txt","a+") as f:
        f.writelines(str(length)+"\n")
    data = []
    with open("length.txt", "r+") as f1:
        for line in f1.readlines():
            line = line.strip("\n")  # 除去换行符
            line = float(line)
            data.append(line)
    length1=int(min(data))
    print(length1)
    raw_sick_data_list_acc = []
    # raw_sick_data_list_wrist = []
    number_of_point_in_each_sample = 100  # 采样窗口大小
    number_count1 = 0
    if length>length1:
        for i in range(number_of_point_in_each_sample):  # 使用重复采样策略，重复采样的次数即为窗口大小
            for j in range(i, length1 - i - number_of_point_in_each_sample, number_of_point_in_each_sample):
                # sick_small_data_batch_gsr = [float(x) for x in all_sick_gsr[j: j + number_of_point_in_each_sample]]
                # sick_small_data_batch_hrt = [float(x) for x in all_sick_hrt[j: j + number_of_point_in_each_sample]]
                sick_small_data_batch_acc = [float(x) for x in all_sick_acc[j: j + number_of_point_in_each_sample]]
                # sick_small_data_batch_wrist = [float(x) for x in all_sick_wrist[j: j + number_of_point_in_each_sample]]
                number_count1 += 1
                # raw_sick_data_list_gsr.append(sick_small_data_batch_gsr)  # 添加到整体数据集中
                # raw_sick_data_list_hrt.append(sick_small_data_batch_hrt)
                raw_sick_data_list_acc.append(sick_small_data_batch_acc)
                # raw_sick_data_list_wrist.append(sick_small_data_batch_wrist)
        for i in range(number_of_point_in_each_sample):  # 使用重复采样0策略，重复采样的次数即为窗口大小
            for j in range(length1, length - i - number_of_point_in_each_sample, number_of_point_in_each_sample):
                # sick_small_data_batch_gsr = [float(x) for x in all_sick_gsr[j: j + number_of_point_in_each_sample]]
                # sick_small_data_batch_hrt = [float(x) for x in all_sick_hrt[j: j + number_of_point_in_each_sample]]
                sick_small_data_batch_acc = [float(x) for x in all_sick_acc[j: j + number_of_point_in_each_sample]]
                # sick_small_data_batch_wrist = [float(x) for x in all_sick_wrist[j: j + number_of_point_in_each_sample]]
                number_count1 += 1
                # raw_sick_data_list_gsr.append(sick_small_data_batch_gsr)  # 添加到整体数据集中
                # raw_sick_data_list_hrt.append(sick_small_data_batch_hrt)
                raw_sick_data_list_acc.append(sick_small_data_batch_acc)
                # raw_sick_data_list_wrist.append(sick_small_data_batch_wrist)
    else:
        for i in range(number_of_point_in_each_sample):  # 使用重复采样策略，重复采样的次数即为窗口大小
            for j in range(i, length - i - number_of_point_in_each_sample, number_of_point_in_each_sample):
                # sick_small_data_batch_gsr = [float(x) for x in all_sick_gsr[j: j + number_of_point_in_each_sample]]
                # sick_small_data_batch_hrt = [float(x) for x in all_sick_hrt[j: j + number_of_point_in_each_sample]]
                sick_small_data_batch_acc = [float(x) for x in all_sick_acc[j: j + number_of_point_in_each_sample]]
                # sick_small_data_batch_wrist = [float(x) for x in all_sick_wrist[j: j + number_of_point_in_each_sample]]
                number_count1 += 1
                # raw_sick_data_list_gsr.append(sick_small_data_batch_gsr)  # 添加到整体数据集中
                # raw_sick_data_list_hrt.append(sick_small_data_batch_hrt)
                raw_sick_data_list_acc.append(sick_small_data_batch_acc)
                # print(raw_sick_data_list_acc)
                # raw_sick_data_list_wrist.append(sick_small_data_batch_wrist)
    return raw_sick_data_list_acc


