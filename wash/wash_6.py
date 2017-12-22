


#!/usr/bin/python
# -*- coding: UTF-8 -*-

#第一步只考虑经纬度影响：先按mall分类，每个mall里对经纬度取mean
#第二步分析wifi信号对店铺的影响
#第三步分析时间（是否放假、几点）对店铺的影响，出图说明


########################	求店铺半径，在knn时加权值	################


import pandas as pd
import numpy as np
import sklearn as sk
#import matplotlib.pyplot as plt  


# 提取
way1='/home/m/桌面/2017.10.24/ali/data/1_use/1_ccf_first_round_shop_info.csv'
way2='/home/m/桌面/2017.10.24/ali/data/1_use/2_ccf_first_round_user_shop_behavior.csv'
way3='/home/m/桌面/2017.10.24/ali/data/1_use/3_evaluation_public.csv'

data1=pd.read_csv(way1)
data2=pd.read_csv(way2)
data3=pd.read_csv(way3)

data=pd.merge(data1,data2,on='shop_id',suffixes=('_real','_variable'))



x=data[['longitude_variable','latitude_variable']]
y=data['shop_id']



#*******************svm太差了，放弃了*****************
#*****************************************************
#*****************************************************
#******************尝试神经网络*************************

from sklearn.preprocessing import StandardScaler
scale=StandardScaler().fit(x)
x1=pd.DataFrame(scale.transform(x))

n=100000
xx,yy=x1[:n],y[:n]



from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier()
mlp.fit(xx,yy)
print(mlp.score(xx,yy))
print('-------------------------')
predict_y=mlp.predict(xx)
rate=float(len(predict_y[predict_y==yy]))/n
print(rate)


'''
n=1万，0.66，0.66
n=2万，0.59，0.59（上次是0.61）
n=3万，0.54，0.54（上次是0.53）
n=5万，0.56，0.56
n=8万，0.48
n=10万，0.41，0.41（上次是0.40）
'''


'''
	#下面是正确率

'''









