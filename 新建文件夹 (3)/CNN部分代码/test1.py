# 田颖 TY–46316
import pandas

account = 13101097823
normal_time = "2020-12-16 10:00:00%2020-12-16 17:00:00&2020-12-16 19:00:00%2020-12-16 23:00:00"
# normal_time="2020-12-16 10:00:00%2020-12-16 10:30:00"#一个半小时以外
sick_time = "2020-12-16 17:57:47%2020-12-16 18:00:00"

# 孙奕辉 SYH-46743
account2 = 15154365413
normal_time2 = "2020-12-18 02:00:00%2020-12-18 10:00:00&2020-12-16 13:00:00%2020-12-16 23:00:00"
# normal_time2="2020-12-16 13:00:00%2020-12-16 13:43:15"
sick_time2 = "2020-12-18 10:41:43%2020-12-18 10:42:37&2020-12-18 10:55:54%2020-12-18 10:56:31"

# # 孙奕辉 SYH-46743
# # account = 15154365413
# # normal_time2 = "2020-12-18 02:00:00%2020-12-18 10:00:00&2020-12-16 13:00:00%2020-12-16 23:00:00"
# normal_time = "2020-12-18 10:11:42%2020-12-18 10:41:42&2020-12-18 10:42:38%2020-12-18 10:55:53"  # 发病前半个小时
# # normal_time = "2020-12-18 10:42:38%2020-12-18 10:55:53&2020-12-18 10:56:32%2020-12-18 11:26:32"  # 发病后半个小时
# sick_time = "2020-12-18 10:41:43%2020-12-18 10:42:37&2020-12-18 10:55:54%2020-12-18 10:56:31"

# 王金柱 WJZ–32952
account3 = 15847173073
normal_time3 = "2020-12-20 04:00:00%2020-12-20 10:00:00&2020-12-20 12:00:00%2020-12-20 15:00:00&2020-12-20 18:00:00%2020-12-20 23:00:00"
# normal_time3="2020-12-20 08:00:00%2020-12-20 08:30:00&2020-12-20 13:00:00%2020-12-20 13:30:00&2020-12-20 19:00:00%2020-12-20 19:30:00"#其余半个小时
sick_time3 = "2020-12-20 10:06:08%2020-12-20 10:07:41&2020-12-20 16:15:59%2020-12-20 16:16:45&2020-12-20 17:15:14%2020-12-20 17:15:50"

# 王江曼 WJM–18003
account4 = 13961327467
normal_time4 = "2020-12-24 12:00:00%2020-12-24 22:00:00&2020-12-25 12:00:00%2020-12-25 20:00:00"
# sick_time4 = "2020-12-24 23:00:35%2020-12-24 23:04:03&2020-12-25 20:38:44%2020-12-25 20:40:45"
# normal_time4="2020-12-24 21:00:00%2020-12-24 21:30:00&2020-12-25 18:00:00%2020-12-25 18:30:00"#其余半个小时
sick_time4 = "2020-12-24 23:00:35%2020-12-24 23:02:35&2020-12-25 20:39:24%2020-12-25 20:40:24"

# 董玉惠 DYH-46815
account5 = 15511931800
normal_time5 = "2020-12-26 01:00:00%2020-12-26 5:00:00&2020-12-26 08:00:00%2020-12-26 16:00:00"
# normal_time5="2020-12-26 04:00:00%2020-12-26 04:30:00"#其余半个小时
sick_time5 = "2020-12-26 06:58:21%2020-12-26 06:59:15"

# 陆艳 LY-46529
account6 = 15820761208
normal_time6 = "2020-12-31 07:00:00%2020-12-31 18:00:00&2021-01-03 02:00:00%2021-01-03 05:00:00"
# normal_time6="2020-12-31 08:00:00%2020-12-31 08:30:00&2021-01-03 03:00:00%2021-01-03 03:30:00"#其余半个小时
sick_time6 = "2020-12-31 06:01:59%2020-12-31 06:02:57&2021-01-03 05:19:00%2021-01-03 05:19:46"

# 孙爽 SS-46701
account7 = 13796889768
normal_time7 = "2020-12-30 01:00:00%2020-12-30 06:00:00&2020-12-30 19:00:00%2020-12-30 23:00:00"
# normal_time7="2020-12-30 04:00:00%2020-12-30 04:30:00&2020-12-30 20:00:00%2020-12-30 20:30:00"#其余半个小时
sick_time7 = "2020-12-30 06:38:04%2020-12-30 06:38:51&2020-12-30 18:03:41%2020-12-30 18:04:25"

# 段怡欣 DYX-46850
account8 = 18732141909
normal_time8 = "2021-01-10 08:00:00%2021-01-10 16:00:00"
# normal_time8 = "2021-01-10 14:00:00%2021-01-10 14:30:00&2021-01-10 22:00:00%2021-01-10 22:30:00&2021-01-11 01:00:00%2021-01-11 01:30:00"#其余半个小时！！！！！！！！！
sick_time8 = "2021-01-10 16:07:24%2021-01-10 16:07:39&2021-01-10 19:51:19%2021-01-10 19:52:50&2021-01-11 03:12:02%2021-01-11 03:13:42"

# 秦硕 QS-46883
account9 = 13930692558
normal_time9 = "2021-01-24 02:00:00%2021-01-24 21:00:00"
# normal_time9="2021-01-24 20:00:00%2021-01-24 20:30:00"#其余半个小时
sick_time9 = "2021-01-24 22:59:04%2021-01-24 23:02:18"

# LHR LHR-47347
account10 = 17637907651
normal_time10 = "2021-02-24 01:00:00%2021-02-24 07:00:00"
# normal_time10="2021-02-24 06:00:00%2021-02-24 06:30:00"#其余半个小时
sick_time10 = "2021-02-24 08:11:32%2021-02-24 08:12:44"

# 范云珍 FYZ-47345
account11 = 15558396067
normal_time11 = "2021-03-12 01:00:00%2021-03-12 14:00:00"
# normal_time11="2021-03-12 12:00:00%2021-03-12 12:30:00"#其余半个小时
sick_time11 = "2021-03-12 14:59:28%2021-03-12 15:00:04"

# 贺晓丽 HXL-48229
account12 = 15805363569
normal_time12 = "2021-04-22 00:50:00%2021-04-22 12:00:00"
# normal_time12="2021-04-22 03:00:00%2021-04-22 03:30:00"#其余半个小时
sick_time12 = "2021-04-22 00:31:46%2021-04-22 00:32:54"

# 杜若翔 DRX-40426
account13 = 18638113708
normal_time13 = "2021-04-30 04:00:00%2021-04-30 18:00:00&2021-05-01 05:00:00%2021-05-01 08:00:00&2021-05-01 09:00:00%2021-05-01 10:00:00"
# normal_time13="2021-04-30 16:00:00%2021-04-30 16:30:00&2021-05-01 06:00:00%2021-05-01 06:30:00&2021-05-01 09:00:00%2021-05-01 09:30:00"#其余半个小时
sick_time13 = "2021-04-30 18:43:31%2021-04-30 18:44:34&2021-05-01 03:58:22%2021-05-01 04:00:27&2021-05-01 19:57:17%2021-05-01 19:59:10"

# XYP XYP-45924
account14 = 13474285782
normal_time14 = "2021-05-20 12:00:00%2021-05-20 22:00:00"
# normal_time14="2021-05-20 05:00:00%2021-05-20 05:30:00&2021-05-20 13:00:00%2021-05-20 13:30:00"#其余半个小时！！！！！！！！！！！
sick_time14 = "2021-05-20 07:31:48%2021-05-20 07:31:58&2021-05-20 10:39:27%2021-05-20 10:40:13"

# 郭文秀 GWX-48491
account15 = 15036067997
normal_time15 = "2021-05-30 06:00:00%2021-05-30 22:00:00"
# normal_time15="2021-05-30 07:00:00%2021-05-30 07:30:00"#其余半个小时
sick_time15 = "2021-05-30 04:38:39%2021-05-30 04:40:15"

# 季琪珂 JQK-50436
account16 = 13921976212
normal_time16 = "2021-10-20 00:00:00%2021-10-20 03:00:00"
# normal_time16="2021-10-20 01:00:00%2021-10-20 01:30:00"#其余半个小时
sick_time16 = "2021-10-20 03:28:57%2021-10-20 03:30:31"

# 常亮 CL-50629
account17 = 13889886092
normal_time17 = "2021-10-23 03:00:00%2021-10-23 19:00:00"
# normal_time17="2021-10-23 18:00:00%2021-10-23 18:30:00"#其余半个小时
sick_time17 = "2021-10-23 20:41:28%2021-10-23 20:43:01"

import Function01
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.impute import SimpleImputer
if __name__ == '__main__':
    # list=[[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
    # list1=[[2,2,2],[3,3,3],[4,4,4],[5,5,5]]
    # list=np.array(list)
    # list1=np.array(list1)
    # print(type(list))
    # print(list)
    # # list=list.tolist()
    # # print(type(list))
    # # print(list)
    # print(type(list[:,1]))
    # x=list[:,1]
    # y=list1[:,1]
    # z=x-y
    # print(z)
    # print(max(abs(z)))
    # print(sum(z))
    #
    # #标准化
    # feature_sick_half_10=Function01.readData("all_feature_10/normal_10_2",91)
    # feature_normal_10=Function01.readData("all_feature_10/normal_10",91)
    #
    # # scaler=StandardScaler()
    # # feature_sick_half_10=scaler.fit_transform(feature_sick_half_10)
    # # feature_normal_10=scaler.fit_transform(feature_normal_10)
    # # print(type(feature_normal_10))
    # # #归一化到-1~1
    # # scaler=MinMaxScaler(feature_range=[-1,1])
    # # feature_sick_half_10=scaler.fit_transform(feature_sick_half_10)
    # # feature_normal_10=scaler.fit_transform(feature_normal_10)
    #将空值置为0
    # feature_normal_10=np.array(feature_normal_10)
    # feature_sick_half_10=np.array(feature_sick_half_10)
    # feature_sick_half_10[np.isnan(feature_sick_half_10)]=0
    # feature_normal_10[np.isnan(feature_normal_10)]=0
    # for i in range(0, len(feature_normal_10)):
    #     for j in range(0,len(feature_normal_10[i])):
    #
    #         with open("all_feature_10/1_normal", "a+") as f2:
    #             if j == 90:
    #                 f2.writelines(str(feature_normal_10[i][j]) + "\n")
    #             else:
    #                 f2.writelines(str(feature_normal_10[i][j]) + ",")
    # print(np.var(feature_normal_10))
    # print(np.var(feature_sick_half_10))
    # print(scaler.fit(feature_normal_10).var_)
    # print(scaler.fit(feature_sick_half_10).var_)

    # from statsmodels.stats.diagnostic import lilliefors
    # from scipy import stats
    # fVal, pSD=stats.levene(feature_normal_10,feature_sick_half_10,center='mean')
    # # lilliefors(feature_normal_10)

    # value_num=0
    # max_value=len(feature_normal_10)*91*2
    # feature_sick_half_10=np.array(feature_sick_half_10)
    # feature_normal_10=np.array(feature_normal_10)
    # for i in range(91):
    #     feature_sick_half_10_value=feature_sick_half_10[:,i]
    #     feature_normal_10_value=feature_normal_10[:,i]
    #     # feature_normal_10_value_max=max(abs(feature_normal_10_value))
    #     # feature_sick_half_10_value_max=max(abs(feature_sick_half_10_value))
    #     # if feature_normal_10_value_max>feature_sick_half_10_value_max:
    #     #     max_value+=feature_normal_10_value_max
    #     # else:
    #     #     max_value+=feature_sick_half_10_value_max
    #     value=feature_normal_10_value-feature_sick_half_10_value
    #     value=sum(abs(value))
    #     value_num+=value
    # print(value_num)
    # print(max_value)
    # print(value_num/max_value)
    #
    # #归一化
    # feature_sick_half_10 = Function01.readData("all_feature_10/normal_sick_10", 91)
    # feature_normal_10 = Function01.readData("all_feature_10/normal_10", 91)
    # scaler=MinMaxScaler()
    # scaler = StandardScaler()
    # feature_sick_half_10 = scaler.fit_transform(feature_sick_half_10)
    # feature_normal_10 = scaler.fit_transform(feature_normal_10)
    # # feature[np.isnan(feature)] = 0
    # # test_fea[np.isnan(test_fea)] = 0
    # feature_sick_half_10[np.isnan(feature_sick_half_10)] = 0
    # feature_normal_10[np.isnan(feature_normal_10)] = 0
    # value_num = 0
    # max_value = 0
    # feature_sick_half_10 = np.array(feature_sick_half_10)
    # feature_normal_10 = np.array(feature_normal_10)
    # for i in range(91):
    #     feature_sick_half_10_value = feature_sick_half_10[:, i]
    #     feature_normal_10_value = feature_normal_10[:, i]
    #     feature_normal_10_value_max = max(abs(feature_normal_10_value))
    #     feature_sick_half_10_value_max = max(abs(feature_sick_half_10_value))
    #     if feature_normal_10_value_max > feature_sick_half_10_value_max:
    #         max_value += feature_normal_10_value_max
    #     else:
    #         max_value += feature_sick_half_10_value_max
    #     value = feature_normal_10_value - feature_sick_half_10_value
    #     value = abs(sum(value))
    #     value_num += value
    # print(value_num)
    # print(max_value)
    # print(value_num / max_value)
    from sklearn.metrics.pairwise import cosine_similarity
    # v1=[[5,3,4,4],[1,1,1,1]]
    # print(np.var(v1))
    # # v1=np.array(v1).reshape(1,-1)
    # print(v1)
    # v2=[[3,1,2,3],[2,2,2,2]]
    # print(np.var(v2))
    # # v2=np.array(v2).reshape(1,-1)
    # print(cosine_similarity(v1, v2))
    # feature_sick_half_10=Function01.readData("one_all_feature_3/sick_2.13_EDA",30)
    # print(len(feature_sick_half_10))
    # feature_normal_10=Function01.readData("全部病人_T检验/normal",90)
    # feature_sick_half_10=Function01.readData("全部病人_T检验/sick",90)
    # feature_premorbid_half_10=Function01.readData("全部病人_T检验/premorbid",90)
    # feature_sick2_half_10=Function01.readData("全部病人_T检验/sick2",90)
    # # # feature_normal_10 = Function01.readData("one_all_feature_3/sick_2.13_ACC", 30)
    # feature_normal_10=np.array(feature_normal_10)
    # feature_sick_half_10=np.array(feature_sick_half_10)
    # feature_premorbid_half_10=np.array(feature_premorbid_half_10)
    # feature_sick2_half_10=np.array(feature_sick2_half_10)
    # feature_sick_half_10[np.isnan(feature_sick_half_10)]=0
    # feature_normal_10[np.isnan(feature_normal_10)]=0
    # feature_premorbid_half_10[np.isnan(feature_premorbid_half_10)]=0
    # feature_sick2_half_10[np.isnan(feature_sick2_half_10)]=0
    #
    # #标准化，服从正态分布
    # transfer = StandardScaler()
    # feature_normal_10 = transfer.fit_transform(feature_normal_10)
    # feature_sick_half_10 = transfer.transform(feature_sick_half_10)
    # feature_premorbid_half_10=transfer.transform(feature_premorbid_half_10)
    # feature_sick2_half_10=transfer.transform(feature_sick2_half_10)
    # # for i in range(0, len(feature_normal_10)):
    # #     for j in range(0, len(feature_normal_10[i])):
    # #         with open("t_分布检验_2分13秒_EDA/normal_1.36_EDA__", "a+") as f2:
    # #             f2.writelines(str(feature_normal_10[i][j]) + "\n")
    # # for i in range(0, len(feature_sick_half_10)):
    # #     for j in range(0, len(feature_sick_half_10[i])):
    # #         with open("t_分布检验_2分13秒_EDA/premorbid_1.36_EDA__", "a+") as f2:
    # #             f2.writelines(str(feature_sick_half_10[i][j]) + "\n")
    #
    # for i in range(0, len(feature_normal_10)):
    #     for j in range(0, len(feature_normal_10[i])):
    #         if j==89:
    #             with open("全部病人_T检验/normal_1", "a+") as f2:
    #                 f2.writelines(str(feature_normal_10[i][j]) + "\n")
    #         else:
    #             with open("全部病人_T检验/normal_1", "a+") as f2:
    #                 f2.writelines(str(feature_normal_10[i][j]) + ",")
    # for i in range(0, len(feature_sick_half_10)):
    #     for j in range(0, len(feature_sick_half_10[i])):
    #         if j==89:
    #             with open("全部病人_T检验/sick_1", "a+") as f2:
    #                 f2.writelines(str(feature_sick_half_10[i][j]) + "\n")
    #         else:
    #             with open("全部病人_T检验/sick_1", "a+") as f2:
    #                 f2.writelines(str(feature_sick_half_10[i][j]) + ",")
    # for i in range(0, len(feature_premorbid_half_10)):
    #     for j in range(0, len(feature_premorbid_half_10[i])):
    #         if j==89:
    #             with open("全部病人_T检验/premorbid_1", "a+") as f2:
    #                 f2.writelines(str(feature_premorbid_half_10[i][j]) + "\n")
    #         else:
    #             with open("全部病人_T检验/premorbid_1", "a+") as f2:
    #                 f2.writelines(str(feature_premorbid_half_10[i][j]) + ",")
    #
    # for i in range(0, len(feature_sick2_half_10)):
    #     for j in range(0, len(feature_sick2_half_10[i])):
    #         if j==89:
    #             with open("全部病人_T检验/sick2_2", "a+") as f2:
    #                 f2.writelines(str(feature_sick2_half_10[i][j]) + "\n")
    #         else:
    #             with open("全部病人_T检验/sick2_2", "a+") as f2:
    #                 f2.writelines(str(feature_sick2_half_10[i][j]) + ",")
    # # print(np.std(feature_sick_half_10)/np.std(feature_normal_10))
    # # print(np.std(feature_sick_half_10))
    # # #归一化到-1~1
    # # scaler=MinMaxScaler(feature_range=[0,1])
    # # feature_sick_half_10=scaler.fit_transform(feature_sick_half_10)
    # # feature_normal_10=scaler.fit_transform(feature_normal_10)
    # #
    # #
    # # def simlarityCalu(vector1, vector2):
    # #     vector1Mod = np.sqrt(vector1.dot(vector1))  # dot表示点积 sqrt开方
    # #     # vector1Mod = np.sqrt(np.dot(vector1.all(),vector1.all()))
    # #     vector2Mod = np.sqrt(vector2.dot(vector2))
    # #     # vector2Mod = np.sqrt(np.dot(vector2.all(),vector2))
    # #     if vector2Mod.all() != 0 and vector1Mod.all() != 0:
    # #         # 求余弦相识度方法（就是向量夹角的余弦）
    # #         simlarity = (vector1.dot(vector2)) / (vector1Mod * vector2Mod)
    # #     else:
    # #         simlarity = 0
    # #     return simlarity
    # from scipy.stats import pearsonr
    #
    # result=cosine_similarity(feature_normal_10, feature_sick_half_10)
    # # result=pearsonr(feature_sick_half_10,feature_normal_10)
    # list1=[]
    # h=len(result)
    # print(h)
    # for i in range(h):
    #    list1.append(sum(result[i])/h)
    # print(list1)
    # # print(list1.index(0.491629405))
    # print(sum(list1) / len(list1))
    # # print(result)
    # # pd1=pandas.DataFrame(result)
    # # feature_normal_10=feature_normal_10[:30]
    # # print(feature_normal_10)
    # # print(len(feature_normal_10))
    # # feature_sick_half_10=feature_sick_half_10[:30]
    # # print(np.array(feature_normal_10).shape)
    # # feature_sick_half_10=feature_sick_half_10[:30]
    # # print(simlarityCalu(np.array(feature_normal_10), np.array(feature_sick_half_10)))
    # # # pandas.DataFrame.to_excel(pd1)
    # # pd1.to_excel("ACC.xlsx")
    # print()
    # print()
    # from scipy.stats import pearsonr
    #
    # i = [1, 0, 0, 0]
    # j = [1, 0.5, 0.5, 0]
    # print(pearsonr(i, j))
    #
    # import pandas as pd
    #
    # dataDf1 = pd.DataFrame({'列1_left': ['a', 'b', 'b', 'c'],
    #                         '列2_left': [1, 2, 2, 3]})
    # dataDf2 = pd.DataFrame({'列1_right': ['b', 'c', 'c', 'd'],
    #                         '列2_right': [2, 3, 3, 4]})
    # print(dataDf1)
    # print(dataDf2)
    #
    # dataLfDf=pd.merge(dataDf1,dataDf2, how='outer')
    # print(dataLfDf)

    x=[[1,2,3]]
    y=[[4,5,6]]
    s = [[4, 5, 7]]
    t = [[4, 5, 8]]
    z=x+y+s+[1]+t
    print(z)




