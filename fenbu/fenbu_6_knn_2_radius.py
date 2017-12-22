
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import sklearn as sk
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

way_write='C:/Users/Administrator/Desktop/ali/data/4_knn/'
train=pd.read_csv(way_write+'train.csv')
mall_list=pd.read_csv(way_write+'mall_list.csv',header=None)[0]

parameters=pd.DataFrame(columns=['loop_number','mall_id','correct_rate','member'])

list=[0.03]
jilu=pd.Series([np.nan]*len(list),index=list)

for j in list:
	for i in range(len(mall_list)):
		x_xuanze = train[train['mall_id']==mall_list[i]][['longitude_variable','latitude_variable']]
		y_xuanze = train[train['mall_id']==mall_list[i]]['shop_id']
		
		x3,y3=x_xuanze,y_xuanze		
		
		#标准化及其保存
		scale=StandardScaler()
		x4=scale.fit_transform(x3)
		y4=y3
		way_tempt_scale='C:/Users/Administrator/Desktop/ali/data/4_knn/scale_2/scale_2_'+mall_list[i]+'.model'
		#joblib.dump(scale,way_tempt_scale)
	
		#切分
		x5,x6,y5,y6=train_test_split(x4, y4, test_size=0.5)
	
		#重新训练
		#knn=KNeighborsClassifier(n_neighbors=len(y3.unique())/3)
		knn=RadiusNeighborsClassifier(radius=j)		#####################
		knn.fit(x5,y5)
		score=knn.score(x6,y6)
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))	
		print(str(i)+'....mall_id='+mall_list[i]+'....'+str(score)+'....'+str(len(x4)))	
		way_tempt_knn='C:/Users/Administrator/Desktop/ali/data/4_knn/scale_2/knn_'+mall_list[i]+'.model'
		#joblib.dump(knn,way_tempt_knn)
		
		#保存
		parameters_append=pd.DataFrame([[i,mall_list[i],score,len(y6)]],columns=['loop_number','mall_id','correct_rate','member'])
		parameters=parameters.append(parameters_append,ignore_index=True)
		#parameters.to_csv(way_write+'parameters.csv')
	
		#print('********************************************************************************')

	score_all=sum(parameters['correct_rate']*parameters['member'])/sum(parameters['member'])
	print(str(score_all)+'....n_neighbors='+str(j))
	
	jilu[j]=score_all
	
print(jilu)





'''
default
3     0.688686
5     0.693253
7     0.695454
10    0.696238
15    0.695662
20    0.694179


weights='distance'
3     0.693660
5     0.694540
7     0.695635
10    0.696851
15    0.697917
20    0.698809


algorithm='ball_tree'
3     0.698016
5     0.698029
7     0.698163
10    0.698176
15    0.697879
20    0.697265

algorithm='kd_tree'
3     0.696850
5     0.696926
7     0.697089
10    0.697119
15    0.696950
20    0.696538

algorithm='brute'
memory error


weights='distance',algorithm='kd_tree'
3     0.691129
5     0.695848
7     0.699128
10    0.701407
15    0.702948
20    0.703710




'''





