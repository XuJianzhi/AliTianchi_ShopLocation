

#!/usr/bin/python
# -*- coding: UTF-8 -*-

#训练	svc	0.700619
#训练	sgd	0.193008864359
#训练	knn	0.723031
#训练	Gaussian	ValueError: array is too big; 
#训练	DecisionTreeClassifier	  0.724201 (30、3)
#训练	GaussianNB	0.258237163405
#训练	MultinomialNB 	0.144171
#训练	MultinomialNB 	0.144171
#训练	BernoulliNB 	 0.286001
#训练	AdaBoostClassifier	0.730222
#训练	ExtraTreesClassifier	0.742766
#训练	GradientBoostingClassifier	运行不了
#训练	RandomForestClassifier	0.725707
#训练	MLPClassifier	0.540893125941


#达到0.70的有：
#训练	svc		0.700619
#训练	knn		0.723031
#训练	DecisionTreeClassifier	  	0.724201 
#训练	AdaBoostClassifier(KDT)		0.730222
#训练	ExtraTreesClassifier		0.742766
#训练	RandomForestClassifier		0.725707



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

way_write='C:/Users/Administrator/Desktop/ali/data/3_tempt/'
x=pd.read_csv(way_write+'x.csv',index_col=0)	#以列标叫0的列作为index
y=pd.read_csv(way_write+'y.csv',index_col=0,header=None)	#否则会把第一行作为列表

#切分
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2)


#标准化及其保存
scale=StandardScaler()
x_train_scaled=scale.fit_transform(x_train)
way_tempt_scale='C:/Users/Administrator/Desktop/ali/data/3_tempt/scale_2.model'
joblib.dump(scale,way_tempt_scale)

x_test_scaled=scale.transform(x_test)


#--------------------------


#训练	svc	0.700619
list1=pd.Series([1,10,100,1000])
list2=list1.copy()
score=pd.DataFrame(pd.Series([np.nan]*len(list1)*len(list2)).reshape(len(list1),len(list2)),index=list1,columns=list2)
max=0
	# 外层（每行）是C，内层（每列）是gamma
for ii in list1:
	for jj in list2:
		clf=SVC(C=ii,gamma=jj,cache_size=8000,decision_function_shape='ovo')
		clf.fit(x_train_scaled,y_train)				
		score.loc[ii,jj]=clf.score(x_test_scaled,y_test)
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print(time_now)
		#记录最大值
		print(score)
		if max<score.loc[ii,jj]:
			max=score.loc[ii,jj]
			ii_max=ii
			jj_max=jj


#训练	sgd	0.193008864359
clf=SGDClassifier(n_jobs=-1)
clf.fit(x_train_scaled,y_train)				
print(clf.score(x_test_scaled,y_test))

#训练	knn	0.723031
list1=pd.Series([1,3,5,10,20])
list2=pd.Series([5,10,20,30,50,70])
score=pd.DataFrame(pd.Series([np.nan]*len(list1)*len(list2)).reshape(len(list1),len(list2)),index=list1,columns=list2)
max=0
	# 外层（每行）是n_neighbors，内层（每列）是leaf_size
for ii in list1:
	for jj in list2:
		clf=KNeighborsClassifier(n_neighbors=ii,leaf_size=jj,n_jobs=-1)
		clf.fit(x_train_scaled,y_train)				
		score.loc[ii,jj]=clf.score(x_test_scaled,y_test)
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print(time_now)
		#记录最大值
		print(score)
		if max<score.loc[ii,jj]:
			max=score.loc[ii,jj]
			ii_max=ii
			jj_max=jj

#训练	Gaussian	ValueError: array is too big; 
list1=pd.Series([0,1,2,3,5,10])
list2=pd.Series([10,30,50,100,150,200])
score=pd.DataFrame(pd.Series([np.nan]*len(list1)*len(list2)).reshape(len(list1),len(list2)),index=list1,columns=list2)
max=0
	# 外层（每行）是n_restarts_optimizer，内层（每列）是max_iter_predict
for ii in list1:
	for jj in list2:
		#clf=GaussianProcessClassifier(n_restarts_optimizer=ii,max_iter_predict=jj,n_jobs=-1)
		clf=GaussianProcessClassifier()
		clf.fit(x_train_scaled,y_train)				
		score.loc[ii,jj]=clf.score(x_test_scaled,y_test)
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print(time_now)
		#记录最大值
		print(score)
		if max<score.loc[ii,jj]:
			max=score.loc[ii,jj]
			ii_max=ii
			jj_max=jj

#训练	DecisionTreeClassifier	  0.724201 (30、3)
list1=pd.Series([2,3,5,10,20,30,40,50])
list2=pd.Series([1,2,3,5,10,20,30,50])
score=pd.DataFrame(pd.Series([np.nan]*len(list1)*len(list2)).reshape(len(list1),len(list2)),index=list1,columns=list2)
max=0
	# 外层（每行）是min_samples_split，内层（每列）是max_iter_predict
for ii in list1:
	for jj in list2:
		clf=DecisionTreeClassifier(min_samples_split=ii,min_samples_leaf=jj)
		clf.fit(x_train_scaled,y_train)				
		score.loc[ii,jj]=clf.score(x_test_scaled,y_test)
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print(time_now)
		#记录最大值
		print(score)
		if max<score.loc[ii,jj]:
			max=score.loc[ii,jj]
			ii_max=ii
			jj_max=jj

#训练	GaussianNB	0.258237163405
clf=GaussianNB()
clf.fit(x_train_scaled,y_train)				
print(clf.score(x_test_scaled,y_test))

#训练	MultinomialNB 	0.144171
list1=pd.Series([1])
list2=pd.Series([0.01,0.1,0.5,1,1.5,2,3,5,10])
score=pd.DataFrame(pd.Series([np.nan]*len(list1)*len(list2)).reshape(len(list1),len(list2)),index=list1,columns=list2)
max=0
	# 
for ii in list1:
	for jj in list2:
		clf=MultinomialNB(alpha=jj)
		clf.fit(x_train_scaled+100,y_train)				
		score.loc[ii,jj]=clf.score(x_test_scaled+100,y_test)
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print(time_now)
		#记录最大值
		print(score)
		if max<score.loc[ii,jj]:
			max=score.loc[ii,jj]
			ii_max=ii
			jj_max=jj

#训练	BernoulliNB 	 0.286001
list1=pd.Series([1])
list2=pd.Series([0.01,0.1,0.5,1,1.5,2,3,5,10])
score=pd.DataFrame(pd.Series([np.nan]*len(list1)*len(list2)).reshape(len(list1),len(list2)),index=list1,columns=list2)
max=0
	#
for ii in list1:
	for jj in list2:
		clf=BernoulliNB(alpha=jj)
		clf.fit(x_train_scaled,y_train)				
		score.loc[ii,jj]=clf.score(x_test_scaled,y_test)
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print(time_now)
		#记录最大值
		print(score)
		if max<score.loc[ii,jj]:
			max=score.loc[ii,jj]
			ii_max=ii
			jj_max=jj
			
			
#训练	AdaBoostClassifier	0.730222

dtc=DecisionTreeClassifier(min_samples_split=30,min_samples_leaf=3)

list1=pd.Series([1,10,30,50,100])
list2=pd.Series([0.01,0.1,0.2,0.5,1,1.5,2,3,5,10])
#list2=pd.Series([0.001,0.01,0.05,0.1,0.2])
score=pd.DataFrame(pd.Series([np.nan]*len(list1)*len(list2)).reshape(len(list1),len(list2)),index=list1,columns=list2)
max=0
	# 外层（每行）是n_estimators，内层（每列）是learning_rate
for ii in list1:
	for jj in list2:
		clf=AdaBoostClassifier(base_estimator=dtc,n_estimators=ii,learning_rate=jj)
		clf.fit(x_train_scaled,y_train)
		score.loc[ii,jj]=clf.score(x_test_scaled,y_test)
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print(time_now)
		#记录最大值
		print(score)
		if max<score.loc[ii,jj]:
			max=score.loc[ii,jj]
			ii_max=ii
			jj_max=jj
			
#训练	ExtraTreesClassifier	0.742766
list1=pd.Series([2,3,5,10,20,30,50,70,100,200])
list2=pd.Series([1,2,3,5,10,20,30,40])
#list2=pd.Series([0.001,0.005,0.01,0.02,0.05,0.08,0.1,0.15,0.2])
score=pd.DataFrame(pd.Series([np.nan]*len(list1)*len(list2)).reshape(len(list1),len(list2)),index=list1,columns=list2)
max=0
	# 外层（每行）是min_samples_split，内层（每列）是n_estimators
for ii in list1:
	for jj in list2:
		clf=ExtraTreesClassifier(min_samples_split=ii,n_estimators =jj)
		clf.fit(x_train_scaled,y_train)				
		score.loc[ii,jj]=clf.score(x_test_scaled,y_test)
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print(time_now)
		#记录最大值
		print(score)
		if max<score.loc[ii,jj]:
			max=score.loc[ii,jj]
			ii_max=ii
			jj_max=jj			
						
#训练	GradientBoostingClassifier	运行不了
list1=pd.Series([2,3,5,10,20,30,50,70,100])
#list2=pd.Series([1,2,3,5,10,20,30,40])
list2=pd.Series([1,2,3,5,10,20,30,50,100])
#list2=pd.Series([0.001,0.005,0.01,0.02,0.05,0.08,0.1,0.15,0.2])
score=pd.DataFrame(pd.Series([np.nan]*len(list1)*len(list2)).reshape(len(list1),len(list2)),index=list1,columns=list2)
max=0
	# 外层（每行）是min_samples_split，内层（每列）是min_samples_leaf
for ii in list1:
	for jj in list2:
		print('***')
		clf=GradientBoostingClassifier(min_samples_split=ii,min_samples_leaf=jj)
		clf.fit(x_train_scaled,y_train)				
		score.loc[ii,jj]=clf.score(x_test_scaled,y_test)
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print(time_now)
		#记录最大值
		print(score)
		if max<score.loc[ii,jj]:
			max=score.loc[ii,jj]
			ii_max=ii
			jj_max=jj			
			
						
#训练	RandomForestClassifier	0.725707
list1=pd.Series([2,3,5,10,20,30,50,70,100])
#list2=pd.Series([1,2,3,5,10,20,30,40])
list2=pd.Series([1,2,3,5,10,20,30,50,100])
#list2=pd.Series([0.001,0.005,0.01,0.02,0.05,0.08,0.1,0.15,0.2])
score=pd.DataFrame(pd.Series([np.nan]*len(list1)*len(list2)).reshape(len(list1),len(list2)),index=list1,columns=list2)
max=0
	# 外层（每行）是min_samples_split，内层（每列）是min_samples_leaf
for ii in list1:
	for jj in list2:
		print('***')
		clf=RandomForestClassifier(min_samples_split=ii,min_samples_leaf=jj)
		clf.fit(x_train_scaled,y_train)				
		score.loc[ii,jj]=clf.score(x_test_scaled,y_test)
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print(time_now)
		#记录最大值
		print(score)
		if max<score.loc[ii,jj]:
			max=score.loc[ii,jj]
			ii_max=ii
			jj_max=jj			
			
			
#训练	MLPClassifier	0.540893125941
clf=MLPClassifier(hidden_layer_sizes=(1000, ),max_iter=1000)
clf.fit(x_train_scaled,y_train)				
print(clf.score(x_test_scaled,y_test))			
			
			
			
			
			
			

