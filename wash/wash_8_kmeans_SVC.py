


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
way1='C:/Users/Administrator/Desktop/ali/data/1_use/1_ccf_first_round_shop_info.csv'
way2='C:/Users/Administrator/Desktop/ali/data/1_use/2_ccf_first_round_user_shop_behavior.csv'
way3='C:/Users/Administrator/Desktop/ali/data/1_use/3_evaluation_public.csv'

data1=pd.read_csv(way1)
data2=pd.read_csv(way2)
data3=pd.read_csv(way3)

data=pd.merge(data1,data2,on='shop_id',suffixes=('_real','_variable'))



x=data[['longitude_variable','latitude_variable']]
y=data['shop_id']


'''
#*******************KMeans：memory error****************
from sklearn.cluster import KMeans
n=10000
xx,yy=x[:n],y[:n]
kmeans = KMeans(n_clusters=8477).fit(xx)
print(kmeans.score(xx,yy))


# 简单
from sklearn.svm import SVC
n=30000
xx,yy=x[:n],y[:n]
svc=SVC(C=1,cache_size=1000,decision_function_shape='ovo')
svc.fit(xx,yy)	#15:16
print('------------')
print(svc.score(xx,yy))
'''
#******************************************

# 复杂

#list1=pd.Series([0.001,0.01,0.1,1,10,100,1000])
list1=pd.Series([0.01,0.1,1,10,100,1000])
#list1=pd.Series([100,1000,10000,100000])

list2=list1.copy()
from sklearn.svm import SVC
import gc

n=80000
xx,yy=x[:n],y[:n]
#xx,yy=x,y

#score=pd.DataFrame(pd.Series([np.nan]*49).reshape(7,7),index=list1,columns=list2)
score=pd.DataFrame(pd.Series([np.nan]*36).reshape(6,6),index=list1,columns=list2)
#score=pd.DataFrame(pd.Series([np.nan]*16).reshape(4,4),index=list1,columns=list2)

# 外层（每行）是C，内层（每列）是gamma
for i in list1:
	for j in list2:
		svc=SVC(C=i
		,gamma=j,cache_size=8000,decision_function_shape='ovo')
		svc.fit(xx,yy)
		print('------------')
		score.loc[i,j]=svc.score(xx,yy)
		print(score)
		#gc.collect()
print(score)	#22:07

'''
n=1万
          0.001     0.010     0.100     1.000     10.000    100.000   1000.000
0.001       0.0824    0.0824    0.1370    0.2468    0.2468    0.3066    0.3062
0.010       0.1370    0.2980    0.3748    0.5406    0.6550    0.6501    0.6499
0.100       0.2980    0.3799    0.5579    0.6599    0.6634    0.6636    0.6635
1.000       0.3799    0.5586    0.6634    0.6639    0.6640    0.6644    0.7078
10.000      0.5586    0.6634    0.6639    0.6640    0.6643    0.7079    0.8155
100.000     0.6634    0.6639    0.6640    0.6643    0.7077    0.8152    0.9008
1000.000    0.6639    0.6640    0.6643    0.7078    0.8040    0.9007    0.9423

n=5wan
          0.001     0.010     0.100     1.000     10.000    100.000   1000.000
0.001      0.02692   0.05120   0.11946   0.16756   0.28306   0.31122   0.31090
0.010      0.07554   0.14522   0.25752   0.47300   0.54364   0.56600   0.57994
0.100      0.14522   0.25752   0.47356   0.54528   0.56758   0.58194   0.58162
1.000      0.25752   0.47356   0.54528   0.56758   0.58190   0.58246   0.64570
10.000     0.47356   0.54528   0.56758   0.58190   0.58226   0.64506   0.77832
100.000    0.54528   0.56758   0.58190   0.58226   0.63734   0.76768   0.88326
1000.000   0.56758   0.58190   0.58226   0.63158   0.76050   0.87162   0.92284

n=5wan
         100      1000     10000    100000
100     0.76768  0.88326  0.92400  0.93698
1000    0.87162  0.92284  0.93398  0.94068
10000   0.91664  0.92982  0.93818  0.94566
100000  0.92722  0.93656  0.94068  0.94706

n=8wan
MemoryError
'''






