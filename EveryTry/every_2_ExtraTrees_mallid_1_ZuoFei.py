

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


#------------------------------




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

way_write='C:/Users/Administrator/Desktop/ali/data/3_tempt/'
x=pd.read_csv(way_write+'x.csv',index_col=0)	#以列标叫0的列作为index
y=pd.read_csv(way_write+'y.csv',index_col=0,header=None)[1]
mall=pd.read_csv(way_write+'mall.csv',index_col=0,header=None)[1]
mall_list=pd.read_csv(way_write+'mall_list.csv',index_col=0,header=None)[1]




x_train,x_test,y_train,y_test=train_test_split(x, mall, test_size=0.2)

#标准化及其保存
scale=StandardScaler()
x_train_scaled=scale.fit_transform(x_train)
way_tempt_scale='C:/Users/Administrator/Desktop/ali/data/3_tempt/mall_test/scale_mall.model'
joblib.dump(scale,way_tempt_scale)

x_test_scaled=scale.transform(x_test)

clf=ExtraTreesClassifier(min_samples_split=2,n_estimators=1,n_jobs=-1)
clf.fit(x_train_scaled,y_train)			
print(clf.score(x_test_scaled,y_test)	)
way_tempt_ETC='C:/Users/Administrator/Desktop/ali/data/3_tempt/mall_test/params_mall.model'

joblib.dump(clf,way_tempt_ETC)












