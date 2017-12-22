aa=Counter(data2_original['user_id'])
aaa=pd.DataFrame(aa.items())

bb=Counter(data3_original['user_id'])
bbb=pd.DataFrame(bb.items())

ccc=pd.merge(aaa,bbb,on=0)

len(aaa)	#	714608
len(bbb)	#	338642
len(ccc)	#	 91335

# 如果不考虑每个手机的定位差异，user_id没有意义

------------------------------------
len(pd.DataFrame(Counter(data1_original['shop_id']).items()))

len(data1['...'].unique())
len(data1['...'].value_counts())
---------------------------------
aa=Counter(data1_original['shop_id'])
aaa=pd.DataFrame(aa.items())

bb=Counter(data2_original['shop_id'])
bbb=pd.DataFrame(bb.items())

'''
cc=Counter(data3_original['shop_id'])
ccc=pd.DataFrame(cc.items())
'''


len(aaa)	#	714608
len(bbb)	#	338642
len(ccc)	#	 91335
------------------------------------------------------------------------------------------------------



#可视化

location=data[['shop_id','longitude_real','latitude_real','longitude_variable','latitude_variable']]





shop_name='s_98832'


x1=location[location['shop_id']==shop_name]['longitude_variable']
y1=location[location['shop_id']==shop_name]['latitude_variable']

x2=location[location['shop_id']==shop_name]['longitude_real']
y2=location[location['shop_id']==shop_name]['latitude_real']


'''
x1=location['longitude_variable']
y1=location['latitude_variable']

x2=location['longitude_real']
y2=location['latitude_real']
'''

fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.scatter(x1,y1)
ax1.scatter(x2,y2)
plt.show()


###
shop=location.groupby(location['shop_id']).size()


###
aaa=pd.DataFrame(np.arange(9).reshape(3,3),index=list('aac'),columns=list('xyz'))
aaa=pd.DataFrame(np.arange(9).reshape(3,3),index=range(2001,2004),columns=list('xyz'))

aac=pd.Index(list('aac'),name='aac')
aaa=pd.DataFrame(np.arange(9).reshape(3,3),index=aac,columns=list('xyz'))


############################
############################

from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)


#改
import pandas as pd
X=pd.DataFrame(X)
y=pd.Series(y)


skf.get_n_splits(X, y)

print(skf)  

for train_index, test_index in skf.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

   
   
##############################内存##########################
import psutil
import os
a=0
for i in range(1000000):
	for j in range(100000):
		a=1+a
		info = psutil.virtual_memory()
		print(info.percent,'-----',a,'---',psutil.Process(os.getpid()).memory_info().rss)
		



#--------------------
#score结果为何为负
x=[	[200,300],
	[200,300],
	[200,300],
	[800,500],
	[800,500],
	[800,500]	]
y=list('aababb')

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2,n_jobs=-1).fit(x)
print(kmeans.score(x,y))
print(kmeans.predict(x))

#--------------------------------------------
import numpy as np
import pandas as pd

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
		#gc.collect() 	# 及时回收内存,要不就爆了
		
	return xx,yy

x=pd.DataFrame(np.arange(22).reshape(11,2))
y=pd.Series([1,1,2,3,3,4,4,5,6,6,7])
a,b=drop_lines(x[5:],y[5:])
print(a)
print(b)


----------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)














