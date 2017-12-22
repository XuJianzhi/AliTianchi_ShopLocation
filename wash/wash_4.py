#!/usr/bin/python
# -*- coding: UTF-8 -*-

#第一步只考虑经纬度影响：先按mall分类，每个mall里对经纬度取mean
#第二步分析wifi信号对店铺的影响
#第三步分析时间（是否放假、几点）对店铺的影响，出图说明


########################	求店铺半径，在knn时加权值	################


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt  


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
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors = 15 , weights='distance')
x=data[['longitude_variable','latitude_variable']]
y=data['shop_id']
clf.fit(x,y)


#示例代码
aaa=pd.DataFrame(np.arange(30).reshape(10,3))
bbb=pd.Series(list('xxxyyyzzzz'))
clf=neighbors.KNeighborsClassifier(n_neighbors=3)
clf.fit(aaa,bbb)
clf.predict(aaa)
'''



######	测试
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10,shuffle=True)




nn=1000000
xx=x[:nn];yy=y[:nn]

for i,j in skf.split(xx,yy):
	print('-------------------------')
	print('Start')	
	x1=xx.iloc[i,:]
	y1=yy[i]
	x2=xx.iloc[j,:]
	y2=yy[j]
	clf = neighbors.KNeighborsClassifier(n_neighbors = 5)
	clf.fit(x1,y1)
	print(clf.score(x2,y2))	
	print('-------------------------')
'''-------------------------
Start
0.733320341048
-------------------------
-------------------------
Start
0.731679344417
-------------------------
-------------------------
Start
0.731764520389
-------------------------
-------------------------
Start
0.734406917478
-------------------------
-------------------------
Start
0.73305883174
-------------------------
-------------------------
Start
0.734342725358
-------------------------
-------------------------
Start
0.731872433589
-------------------------
-------------------------
Start
0.73335904671
-------------------------
-------------------------
Start
0.735163780428
-------------------------
-------------------------
Start
0.733215638375
-------------------------
'''	

	
######	初步做出data3的结果
from sklearn import neighbors

print('-------------------------')
print('Start')	
clf = neighbors.KNeighborsClassifier(n_neighbors = 5)
clf.fit(x,y)
problem1=data3[['longitude','latitude']]
result1=clf.predict(problem1)
print(result1)	
print('-------------------------')

result1_upload=pd.DataFrame()
result1_upload['row_id']=data3['row_id']
result1_upload['shop_id']=pd.Series(result1)
print(result1_upload)
result1_upload.to_csv('C:/Users/Administrator/Desktop/ali/result/20171019/result.csv',index=False)

	
	
	
	
	
	
	
	
'''
fenlei0=pd.DataFrame()	
fenlei1=fenlei0.copy()
k=0
for i,j in skf.split(x,y):
	fenlei0[k]=pd.Series(i)	
	fenlei1[k]=pd.Series(j)
	k=k+1
'''
	
	
	
'''	
clf = neighbors.KNeighborsClassifier(n_neighbors = 5)

nn=10
xx=x[:nn];yy=y[:nn]

fenlei0=pd.DataFrame()	
fenlei1=fenlei0.copy()
k=0
for i,j in skf.split(xx,yy):
	print(type(i));print(j)
	fenlei0[k]=pd.Series(i)	
	fenlei1[k]=pd.Series(j)
	k=k+1
'''
	
	
'''
fenlei0.to_csv('C:/Users/Administrator/Desktop/fenlei0.csv')
fenlei1.to_csv('C:/Users/Administrator/Desktop/fenlei1.csv')
'''




############


from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10,shuffle=True)

nn=50000
xx=x[:nn];yy=y[:nn]

for i,j in skf.split(xx,yy):
	print('-------------------------')
	print('Start')	
	x1=xx.iloc[i,:]
	y1=yy[i]
	x2=xx.iloc[j,:]
	y2=yy[j]
	clf = SVC()
	clf.fit(x1,y1)
	print(clf.score(x2,y2))	
	print('-------------------------')













