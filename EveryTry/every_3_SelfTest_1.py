


#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import sklearn as sk
#import matplotlib.pyplot as plt  
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import *
from sklearn.ensemble import *
from sklearn.neural_network import MLPClassifier
import gc

way_write='C:/Users/Administrator/Desktop/ali/data/3_tempt/'
x=pd.read_csv(way_write+'x.csv',index_col=0)	#以列标叫0的列作为index
y=pd.read_csv(way_write+'y.csv',index_col=0,header=None)[1]
mall=pd.read_csv(way_write+'mall.csv',index_col=0,header=None)[1]
mall_list=pd.read_csv(way_write+'mall_list.csv',index_col=0,header=None)[1]
#sum=0
#sum=103639.000000
sum=225808.000000
#i=17 空缺
for i in range(18,len(mall_list)):
	xx = x[mall==mall_list[i]]
	yy = y[mall==mall_list[i]]

	#标准化及其保存
	way_tempt_scale='C:/Users/Administrator/Desktop/ali/data/3_tempt/ETC/scale_i'+str(i)+'.model'
	scale=joblib.load(way_tempt_scale)
	xx_scaled=scale.transform(xx)
	
	#--------------------------
	
	way_tempt_ETC='C:/Users/Administrator/Desktop/ali/data/3_tempt/ETC/params2_i'+str(i)+'.model'
	clf=joblib.load(way_tempt_ETC)
	score=clf.score(xx_scaled,yy)		
	length=len(xx)
	sum=sum+score*length
	
	print('i=%d, score=%f, length=%d, sum=%f' % (i, score, length, sum))
	gc.collect()
	
print(float(sum)/len(x))		#0.8155613063096708


'''
i==17时
In [4]:     score1=clf.score(xx_scaled[:8000],yy[:8000])

In [5]: score1
Out[5]: 0.732375

In [6]: score2=clf.score(xx_scaled[8000:],yy[8000:])

In [7]: score2
Out[7]: 0.81089016415320969

In [8]: score1*8000+score2*(len(yy)-8000)
Out[8]: 11935.0
'''




