# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:37:57 2020

"""


# -*- coding: utf-8 -*-
import time
start=time.clock()
from sklearn.metrics import mean_absolute_error#平均绝对误差（Mean absolute error）
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
import pandas as pd
from sklearn.feature_selection import SelectKBest ,chi2
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn import metrics
from pyspark.sql import SparkSession # instantiate spark session 
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
   
spark = (    

   
    SparkSession     

   
    .builder     

   
    .getOrCreate()     

   
    ) 

   
sc = spark.sparkContext  


def mean(list_):
    sum=0.0
    L=len(list_)
    for i in list_:
        sum=sum+float(i)
    return sum/L





file_path=r'C:\\Users\\DengBY\\Desktop\\liver_Lasso_features\\ALL_V.csv'
data = pd.read_csv(file_path)
a=pd.DataFrame(data)


X=a.values[:,1:597]
y=a.values[:,598]


min_max_scaler = preprocessing.MinMaxScaler()#范围0-1缩放标准化
X=min_max_scaler.fit_transform(X)
#基于L1的特征选择
dimension=['P_D_LASSO']#,30,35,40,45,50,75,100]

L=len(dimension)

for j in dimension:

    lsvc=LassoCV().fit(X, y)

    model = SelectFromModel(lsvc, prefit=True)
    X_lsvc = model.transform(X)
    df_X_lsvc=pd.DataFrame(X_lsvc)
    feature_names = df_X_lsvc.columns.tolist()#显示列名
#    print (feature_names)

    
    y=pd.DataFrame(y)
    b=df_X_lsvc
    objs=[b,y]
    data=pd.concat(objs, axis=1, join='outer', join_axes=None, ignore_index=False,
                   keys=None, levels=None, names=None, verify_integrity=False)
    
    
    
    mean_score=[]
    max_score=[]
    min_score=[]
    mean_auc=[]
    max_auc=[]
    min_auc=[]
    min_recall=[]
    max_recall=[]
    mean_recall=[]
    min_F1=[]
    max_F1=[]
    mean_F1=[]
    precision_min=[]
    precision_max=[]
    precision_mean=[]
    auc=[]
    score=[]
    precision_wei=[]
    precision_hong=[]
    recall=[]
    F1=[]
    kappa=[]
    Mean_absolute_error=[]
    Mean_squared_error=[]
    Median_absolute_error=[]
#    
#    
#    
#    
    for i in range(1):
        
        data=data.sample(frac=1)
        
#        print (j)
        
        x=(j)
        
 #       print(x)
#        
        Y=(j)
        
#        print (Y)
    
        X=data.values[:,:14]
        
        y=data.values[:,15]
        #print (y)
        
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import classification_report
        from sklearn.svm import SVC
       
        from sklearn.grid_search import GridSearchCV
        from sklearn.metrics import cohen_kappa_score
    
        #加载数据
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=200)
        #构建模型，选择核函数并训练
        clf = SVC()
        clf.set_params(kernel='sigmoid', probability=True).fit(X_train, y_train)
        preds1 = clf.predict(X_test)
        print("基准测试集验证得分:"+str(np.mean(preds1 == y_test)))
        
        #设置即将要变换的参数
        param_grid = {'kernel':['poly','linear','sigmoid','rbf'],'C': [0.1, 1, 10, 100, 1000],'gamma':[1,0.1,0.01,0.001,0.0001]}    
        #构建自动调参容器，n_jobs参数支持同时多个进程运行并行测试
        grid_search = GridSearchCV(clf, param_grid, n_jobs = 1, verbose=10)    
        grid_search.fit(X_train, y_train)
        #选出最优参数    
        best_parameters = grid_search.best_estimator_.get_params()    
#        for para, val in list(best_parameters.items()):    
#            print(para, val)
    #使用最优参数进行训练
        
        clf = SVC(kernel=best_parameters['kernel'], C=best_parameters['C'], gamma=best_parameters['gamma'] ,probability=True).fit(X_train, y_train)
        pred_train=clf.predict(X_train)
        fpr1, tpr1, threshold1 = metrics.roc_curve(y_train, pred_train )
        roc_auc1 = metrics.auc(fpr1, tpr1)
        #auc.append(roc_auc)
        plt.figure(figsize=(6,6))
        plt.title('Training ROC')
        plt.plot(fpr1, tpr1, 'b', label = 'Val AUC = %0.3f' % roc_auc1)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()   
        
#        preds1 = clf.predict(X_test)
#        a=("最优测试集验证得分:"+str(np.mean(preds1 == y_test)))
#        #print (a)
#        SCORE=(np.mean(preds1 == y_test))
#        score.append(SCORE)
#        
#        y_predict=clf.predict(X_test)
#        #print(classification_report(y_test, y_predict))
#        
#        precision=metrics.precision_score(y_test, y_predict, average='micro')  # 微平均，精确率
#    
#        precision_wei.append(precision)
#        
#        precision_=metrics.precision_score(y_test, y_predict, average='macro')  # 宏平均，精确率
#    
#        precision_hong.append(precision_)
#        
#        recall_=metrics.recall_score(y_test, y_predict, average='micro')
#        
#        recall.append(recall_)
#        
#        F1_=metrics.f1_score(y_test, y_predict, average='weighted')
#        
#        F1.append(F1_)
#        
#        KAPPA=cohen_kappa_score(y_test, y_predict)
#        
#        kappa.append(KAPPA)
#        
#        Mean_absolute_error_=mean_absolute_error(y_test, y_predict)
#        
#        Mean_absolute_error.append(Mean_absolute_error_)
#        
#        mean_squared_error_=mean_squared_error(y_test, y_predict)
#        
#        Mean_squared_error.append(mean_squared_error_)
#        
#        Median_absolute_error_=median_absolute_error(y_test, y_predict)
#        
#        Median_absolute_error.append(Median_absolute_error_)
#    
        pred = clf.predict_proba(X_test)[:,1]
        
    #    ############画图部分
        fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
        roc_auc = metrics.auc(fpr, tpr)
        auc.append(roc_auc)
        plt.figure(figsize=(6,6))
        plt.title('Validation ROC')
        plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()    
#
##    score=np.array(score)
##    np.save('C:\\Users\\DengBY\\Desktop\\'+j+'\\score.npy',score) # 保存为.npy格式
##    
##    auc=np.array(auc)
##    np.save('C:\\Users\\DengBY\\Desktop\\'+j+'\\auc.npy',auc) # 保存为.npy格式
##    
##    precision_hong=np.array(precision_hong)
##    np.save('C:\\Users\\DengBY\\Desktop\\'+j+'\\precision_hong.npy',precision_hong) # 保存为.npy格式
##    
##    recall=np.array(recall)
##    np.save('C:\\Users\\DengBY\\Desktop\\'+j+'\\recall.npy',recall) # 保存为.npy格式
##    
##    F1=np.array(F1)
##    np.save('C:\\Users\\DengBY\\Desktop\\'+j+'\\F1.npy',F1) # 保存为.npy格式
##    
##    kappa=np.array(kappa)
##    np.save('C:\\Users\\DengBY\\Desktop\\'+j+'\\kappa.npy',kappa) # 保存为.npy格式
##    
##    Mean_absolute_error=np.array(Mean_absolute_error)
##    np.save('C:\\Users\\DengBY\\Desktop\\'+j+'\\Mean_absolute_error.npy',Mean_absolute_error) # 保存为.npy格式
##    
##    Mean_squared_error=np.array(Mean_squared_error)
##    np.save('C:\\Users\\DengBY\\Desktop\\'+j+'\\Mean_squared_error.npy',Mean_squared_error) # 保存为.npy格式
##    
##    Median_absolute_error=np.array(Median_absolute_error)
##    np.save('C:\\Users\\DengBY\\Desktop\\'+j+'\\Median_absolute_error.npy',Median_absolute_error) # 保存为.npy格式
##end=time.clock()
##print ('执行时间',end-start)