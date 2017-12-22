


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
z=data['mall_id']


#*******************KMeans：memory error****************
from sklearn.cluster import KMeans
import gc
n=100000
#xx,yy,zz=x[:n],y[:n],z[:n]
xx,yy,zz=x,y,z
#kmeans = KMeans(n_clusters=97,n_jobs=-1,max_iter=3000).fit(xx)
kmeans = KMeans(n_clusters=25,n_jobs=-1).fit(xx)
print(kmeans.score(xx,zz))#-19.0829318168
#score怎么出来负数了呢？？是做成回归了吗？怎么是用分类？
#答：把x中的每个value减去同意分类中的所在维度的平均值的平均值后做平方，再把这些平方们做加和。score(x,y)的y根本没用。
fenlei1=kmeans.predict(xx)
fenlei1=pd.Series(fenlei1)
way4='C:/Users/Administrator/Desktop/ali/data/3_tempt/fenlei1.csv'
fenlei1.to_csv(way4)
#fenlei1=pd.read_csv(way4)
way5='C:/Users/Administrator/Desktop/ali/data/3_tempt/params1.csv'
pd.Series(kmeans.get_params()).to_csv(way5)

#svc的y如果有其中一类只有一个样本，就会error，所有要去掉这个
def drop_lines(x,y):
	to_drop=(y.groupby(y).size()==1)
	to_drop=pd.Series(to_drop[to_drop].index)
	xx=x.copy()
	yy=y.copy()
	j=0
	for i in xrange(len(to_drop)):
		kkk= yy!=to_drop[i]
		xx=xx[kkk]
		yy=yy[kkk]
		j+=1
		#gc.collect() 	# 及时回收内存,要不就爆了
	return xx,yy

from sklearn.svm import SVC
for i in range(25):
	print(i)
	xuanze=fenlei1==i
	#svc的y如果有其中一类只有一个样本，就会error，所有要去掉这个
	#x3,y3=drop_lines(xx[xuanze],yy[xuanze])
	x3,y3=xx[xuanze],yy[xuanze]
		
	#实验中，当i=21时仅有一个样本。经drop_lines后为空集。
	if len(x3)==0:	continue
	
	svc=SVC(C=1,cache_size=8000,decision_function_shape='ovo')
	svc.fit(x3,y3)
	print('++++++++++++++++')
	print(svc.score(xx[xuanze],yy[xuanze]))
	way_tempt='C:/Users/Administrator/Desktop/ali/data/3_tempt/params2'+str(i)+'.csv'
	params_tempt=pd.Series(svc.get_params())
	params_tempt.to_csv(way_tempt)
	print('--------------------------')
	gc.collect()
	
'''
# 简单
from sklearn.svm import SVC
n=30000
xx,yy=x[:n],y[:n]
svc=SVC(C=1,cache_size=1000,decision_function_shape='ovo')
svc.fit(xx,yy)	#15:16
print('------------')
print(svc.score(xx,yy))

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
'''






