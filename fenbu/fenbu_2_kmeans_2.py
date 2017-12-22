
'''
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

way_write='C:/Users/Administrator/Desktop/ali/data/3_tempt/'

x.to_csv(way_write+'x.csv')
y.to_csv(way_write+'y.csv')
z.to_csv(way_write+'z.csv')
'''


#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import sklearn as sk
#import matplotlib.pyplot as plt  

way_write='C:/Users/Administrator/Desktop/ali/data/3_tempt/'
x=pd.read_csv(way_write+'x.csv')
#y=pd.read_csv(way_write+'y.csv',header=None)	#否则会把第一行作为列表
z=pd.read_csv(way_write+'z.csv',header=None)

#------------------------------------------------------------------------
#改成了两层kmeans

from sklearn.cluster import KMeans
import gc
xx,zz=x,z
kmeans = KMeans(n_clusters=25,n_jobs=-1).fit(xx)
print(kmeans.score(xx,zz))#-19.0829318168
#score怎么出来负数了呢？？是做成回归了吗？怎么是用分类？
#答：把x中的每个value减去同意分类中的所在维度的平均值的平均值后做平方，再把这些平方们做加和。score(x,y)的y根本没用。
fenlei1=kmeans.predict(xx)
fenlei1=pd.Series(fenlei1)
way4='C:/Users/Administrator/Desktop/ali/data/3_tempt/fenlei1.csv'
fenlei1.to_csv(way4,index=False)
way5='C:/Users/Administrator/Desktop/ali/data/3_tempt/params1.csv'
pd.Series(kmeans.get_params()).to_csv(way5)

#第二层kmeans
for i in range(25):
	print(i)
	xuanze=((fenlei1==i))
	
	#有的样本个数就少于40个（其实只有i为21时，只有三个样本）
	if len(xx[xuanze])<=40:
		kmeans_tempt=KMeans(len(xx[xuanze]),n_jobs=-1).fit(xx[xuanze])
	else:	#绝大部分的样本数量要大于40
		kmeans_tempt=KMeans(n_clusters=40,n_jobs=-1).fit(xx[xuanze])
	print(kmeans.score(xx[xuanze]))
	fenlei_tempt=pd.Series(kmeans_tempt.predict(xx[xuanze]))
	way_fenlei_tempt='C:/Users/Administrator/Desktop/ali/data/3_tempt/second_kmeans/fenlei2_'+str(i)+'.csv'
	fenlei_tempt.to_csv(way_fenlei_tempt,index=False)
	way_params_tempt='C:/Users/Administrator/Desktop/ali/data/3_tempt/second_kmeans/params2_'+str(i)+'.csv'
	pd.Series(kmeans_tempt.get_params()).to_csv(way_params_tempt)







