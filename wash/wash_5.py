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


# 加入scaler
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x_trans=pd.DataFrame(scale.fit_transform(x))
#problem1_1=scale.transform(problem1)





######	测试
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10,shuffle=True)


nn=1000000
xx=x_trans[:nn];yy=y[:nn]

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
	
'''
未做scale时：
nn=1000000：

-------------------------
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




做了scale时：（几乎无变化）
nn=1000000：
-------------------------
Start
0.731060089267
-------------------------
-------------------------
Start
0.731195969655
-------------------------
-------------------------
Start
0.73365402624
-------------------------
-------------------------
Start
0.731872205866
-------------------------
-------------------------
Start
0.732480583831
-------------------------
-------------------------
Start
0.73388698321
-------------------------
-------------------------
Start
0.733223033776
-------------------------
-------------------------
Start
0.735083071989
-------------------------
-------------------------
Start
0.733634841555
-------------------------
-------------------------
Start
0.734495288539
-------------------------


做了scale，且nn=30万时：
score为0.823
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




#######	第二种方法：SVC

#此是原始测试方式
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import sklearn

from sklearn.preprocessing import StandardScaler
scale=StandardScaler().fit(x)
x=pd.DataFrame(scale.transform(x))




skf = StratifiedKFold(n_splits=4,shuffle=True)

nn=50000
xx=x[:nn];yy=y[:nn]

for i,j in skf.split(xx,yy):
	print('-------------------------')
	print('Start')	
	x1=xx.iloc[i,:]
	y1=yy[i]
	x2=xx.iloc[j,:]
	y2=yy[j]
	clf = SVC(C=0.01)
	clf.fit(x1,y1)
	print(clf.score(x2,y2))	
	#下面是正确率
	predict_y2=clf.predict(x2)
	rate=float(len(predict_y2[predict_y2==y2]))/nn
	print(rate)
	print('-------------------------')



#此是进化测试方式，出正确率的方程
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=4,shuffle=True)

from sklearn.preprocessing import StandardScaler
scale=StandardScaler().fit(x)
x=pd.DataFrame(scale.transform(x))


nn=10000
xx=x[:nn];yy=y[:nn]
chuan1=pd.Series([0.1,1,10,100,1000,10000,100000])	#gamma
chuan2=chuan1.copy()	#C

jieguo1=pd.DataFrame([[np.nan]*7]*7,index=chuan1,columns=chuan2)
jieguo2=jieguo1.copy()

for q in chuan1:	#gamma
	for p in chuan2:	#C
		for i,j in skf.split(xx,yy):
			print('-------------------------')
			print('Start')	
			x1=xx.iloc[i,:]
			y1=yy[i]
			x2=xx.iloc[j,:]
			y2=yy[j]
			clf = SVC(gamma=q,C=p)
			clf.fit(x1,y1)
			#print(clf.score(x2,y2))	
			#下面是正确率
			predict_y2=clf.predict(x2)
			rate=float(len(predict_y2[predict_y2==y2]))/nn
			#print(rate)
			print('-------------------------')
		jieguo1.loc[q,p]=clf.score(x2,y2)
		jieguo2.loc[q,p]=rate

print(jieguo1)
print('+++++++++++++')
print(jieguo2)
'''
          0.001     0.010     0.100     1.000     10.000    100.000   1000.000
0.001     0.083098  0.137959  0.299718  0.381605  0.546188  0.662364  0.665994
0.010     0.083098  0.299718  0.381605  0.546188  0.662767  0.665994  0.665591
0.100     0.083098  0.376765  0.545785  0.661960  0.666398  0.666398  0.667205
1.000     0.083098  0.538524  0.661557  0.666398  0.666398  0.664784  0.709560
10.000    0.083098  0.633320  0.665591  0.665994  0.665591  0.707543  0.786204
100.000   0.083098  0.653489  0.665188  0.665188  0.707543  0.803953  0.896329
1000.000  0.083098  0.653489  0.663574  0.708754  0.799113  0.898346  0.930617
+++++++++++++
          0.001     0.010     0.100     1.000     10.000    100.000   1000.000
0.001       0.0206    0.0342    0.0743    0.0946    0.1354    0.1642    0.1651
0.010       0.0206    0.0743    0.0946    0.1354    0.1643    0.1651    0.1650
0.100       0.0206    0.0934    0.1353    0.1641    0.1652    0.1652    0.1654
1.000       0.0206    0.1335    0.1640    0.1652    0.1652    0.1648    0.1759
10.000      0.0206    0.1570    0.1650    0.1651    0.1650    0.1754    0.1949
100.000     0.0206    0.1620    0.1649    0.1649    0.1754    0.1993    0.2222
1000.000    0.0206    0.1620    0.1645    0.1757    0.1981    0.2227    0.2307
'''

	
	
'''
nn=3万，n_splits=10，未做scale时：0.55
nn=3万，n_splits=4，未做scale时：0.55
nn=5万，n_splits=4，未做scale时：0.56
nn=所有，n_splits=5，未做scale时：

'''






########	使用  grid  &  SVC	##########

#	预处理：使用SVC时，如果样本中label有的class只有一个样本，SVC在fit时会出error。故应先删去所有此类样本。

'''
rr=pd.Series(list('aabccdeee'),index=list('qwertyuio'))
tt=pd.DataFrame(np.arange(18).reshape(9,2),index=list('qwertyuio'),columns=list('nm'))

rr=pd.Series(list('aabccdeee'))
tt=pd.DataFrame(np.arange(18).reshape(9,2))

x,y=tt,rr
a,b=drop_lines(tt,rr)

nn=800000
xx=x[:nn];yy=y[:nn]
'''


'''
def drop_lines(x,y):
	to_drop=(y.groupby(y).size()==1)
	to_drop=pd.Series(to_drop[to_drop].index)
	xx=x.copy()
	yy=y.copy()
	
	j=0
	
	for i in to_drop:
		print(j,'--------',i)
		kkk= yy!=i
		xx=xx[kkk]
		yy=yy[kkk]
		j+=1
		
	return xx,yy
'''	
	
	
	
	
	


import psutil
import os
import gc

def drop_lines(x,y):
	to_drop=(y.groupby(y).size()==1)
	to_drop=pd.Series(to_drop[to_drop].index)
	xx=x.copy()
	yy=y.copy()
	
	j=0
	
	for i in xrange(len(to_drop)):
		'''
		info = psutil.virtual_memory()
		print(j,'----1----',to_drop[i],'----',info.percent)	# 查看内存使用比例
		'''
		kkk= yy!=to_drop[i]
		xx=xx[kkk]
		yy=yy[kkk]
		j+=1
		gc.collect() 	# 及时回收内存,要不就爆了
		
	return xx,yy
	
x_droped,y_droped=drop_lines(x,y)




import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt  

#	经drop_lines()函数已处理好的
x_droped=pd.read_csv('C:/Users/Administrator/Desktop/x_droped.csv')	
y_droped=pd.read_csv('C:/Users/Administrator/Desktop/y_droped.csv')	


#print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

'''
nn=len(x)
#nn=10002
xx=x_droped[:nn];yy=y_droped[:nn]
'''

n=10000

xx,yy=x_droped[:n],y_droped[:n]

scaler = StandardScaler()
X = scaler.fit_transform(xx)

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, yy)

print("The best parameters are %s with a score of %0.2f"	% (grid.best_params_, grid.best_score_))




#*******************svm太差了，放弃了*****************
#*****************************************************
#*****************************************************
#******************尝试神经网络*************************

from sklearn.preprocessing import StandardScaler
scale=StandardScaler().fit(x)
x1=pd.DataFrame(scale.transform(x))

n=80000
xx,yy=x1[:n],y[:n]

from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier()
mlp.fit(xx,yy)
print(mlp.score(xx,yy))

'''
n=1万，0.66
n=2万，0.61
n=3万，0.55
n=5万，0.56
n=8万，error
n=9万，error
'''




	#下面是正确率
	predict_y2=clf.predict(x2)
	rate=float(len(predict_y2[predict_y2==y2]))/nn
	print(rate)
	print('-------------------------')











