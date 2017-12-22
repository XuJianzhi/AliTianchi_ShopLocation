


#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import sklearn as sk
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
mall_list=pd.read_csv(way_write+'mall_list.csv',index_col=0,header=None)[1]

way3='C:/Users/Administrator/Desktop/ali/data/1_use/3_evaluation_public.csv'
data3=pd.read_csv(way3)
data3=data3.set_index('row_id')

x0=data3[['longitude','latitude']]
mall=data3['mall_id']

way_write='C:/Users/Administrator/Desktop/ali/data/3_tempt/'
scale0=joblib.load(way_write+'scale_1.model')
x=scale0.transform(x0)

result=pd.Series()

for i in range(len(mall_list)):
	xx = x[mall==mall_list[i]]

	#标准化
	way_tempt_scale='C:/Users/Administrator/Desktop/ali/data/3_tempt/ETC/scale_i'+str(i)+'.model'
	scale=joblib.load(way_tempt_scale)
	xx_scaled=scale.transform(xx)
	
	#--------------------------
	
	way_tempt_ETC='C:/Users/Administrator/Desktop/ali/data/3_tempt/ETC/params2_i'+str(i)+'.model'
	clf=joblib.load(way_tempt_ETC)
	result_part=pd.Series(clf.predict(xx_scaled),index=mall[mall==mall_list[i]].index)
	result=pd.concat([result,result_part])
	print('i=%d' % (i))
	gc.collect()
	

result.index.name='row_id'
result_write=result[data3.index]
result_write=pd.DataFrame(result_write,columns=['shop_id'])

way_result='C:/Users/Administrator/Desktop/ali/data/3_tempt/result/result.csv'
result_write.to_csv(way_result)



