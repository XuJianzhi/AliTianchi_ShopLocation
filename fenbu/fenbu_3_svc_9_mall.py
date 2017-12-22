

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

way_write='C:/Users/Administrator/Desktop/ali/data/3_tempt/'
train=pd.read_csv(way_write+'train.csv')

mall_list=train['mall_id'].unique()
number_loop=0
for i in mall_list:
	print(i+'....number_loop='+str(number_loop))
	x_xuanze = train[train['mall_id']==i][['longitude_variable','latitude_variable']]
	y_xuanze = train[train['mall_id']==i]['shop_id']
	'''
	#报错原因之二：实验中，有时仅有一个样本。经drop_lines后为空集。
	if len(x_xuanze)==0:
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print('i='+str(i)+',j='+str(j)+',,,,'+time_now+',,,,'+'situation 2')
		continue
		
	#报错原因之〇：每行的label都不同
	if len(y_xuanze)==len(y_xuanze[1].unique()):
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print('i='+str(i)+',j='+str(j)+',,,,'+time_now+',,,,'+'situation 0')
		x3,y3=x_xuanze.iloc[0,:],y_xuanze.iloc[0,:]
	#报错原因之三：所有样本有一个共同的label	（其实是因为懒了，直接把后面的搬到这里承接“原因之〇”）
	#if len(y3.unique())==1:
		just_one_label_append=pd.DataFrame([[i,j,y3.unique()[0]]],columns=['i','j','label'])
		just_one_label=just_one_label.append(just_one_label_append,ignore_index=True)
		print('i='+str(i)+',j='+str(j)+',,,,'+'just_one_label')
		
		#保存只有一种label的i和j
		way_tempt_2='C:/Users/Administrator/Desktop/ali/data/3_tempt/svc/same_label.scv'
		just_one_label.to_csv(way_tempt_2)
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print('i='+str(i)+',j='+str(j)+',,,,'+time_now+',,,,'+'situation 3')
		continue		
	
	
	#报错原因之一：svc的y如果有其中一类只有一个样本，就会error，所有要去掉这个
	x3,y3=drop_lines(x_xuanze,y_xuanze[1])
	#x3,y3=xx.loc[xuanze[xuanze].index,:],yy.loc[xuanze[xuanze].index,:][1]
	#??????????????????原先的版本为何没出问题?????????????????????????????????????????????????????????????????????????????????????????????????????
	
	#报错原因之二：实验中，有时仅有一个样本。经drop_lines后为空集。
	if len(x3)==0:
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print('i='+str(i)+',j='+str(j)+',,,,'+time_now+',,,,'+'situation 2')
		continue

	#报错原因之三：所有样本有一个共同的label
	if len(y3.unique())==1:
		just_one_label_append=pd.DataFrame([[i,j,y3.unique()[0]]],columns=['i','j','label'])
		just_one_label=just_one_label.append(just_one_label_append,ignore_index=True)
		print('i='+str(i)+',j='+str(j)+',,,,'+'just_one_label')
		
		#保存只有一种label的i和j
		way_tempt_2='C:/Users/Administrator/Desktop/ali/data/3_tempt/svc/same_label.scv'
		just_one_label.to_csv(way_tempt_2)
		print('------------------------------------------------------------------------------')
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print('i='+str(i)+',j='+str(j)+',,,,'+time_now+',,,,'+'situation 3')
		continue
	'''
	
	x3,y3=x_xuanze,y_xuanze			####
	
	#标准化及其保存
	scale=StandardScaler()
	x4=scale.fit_transform(x3)
	y4=y3
	way_tempt_scale='C:/Users/Administrator/Desktop/ali/data/3_tempt/svc/scale_'+i+'.model'
	joblib.dump(scale,way_tempt_scale)

	#list1=pd.Series([0.001,0.01,0.1,1,10,100,1000])
	list1=pd.Series([0.01,0.1,1,10,100,1000])
	#list1=pd.Series([0.001,0.1,10,1000])
	#list1=pd.Series([1,10,100,1000,10000])
	#list1=pd.Series([100,1000,10000,100000])
	#list1=pd.Series([20,40,60,80,100])
	
	list2=list1.copy()
	#list2=pd.Series([10,100])
	
	score=pd.DataFrame(pd.Series([np.nan]*len(list1)*len(list2)).reshape(len(list1),len(list2)),index=list1,columns=list2)
	
	#切分
	x5,x6,y5,y6=train_test_split(x4, y4, test_size=0.2)
	'''
	#防止切分后的训练集y5全是相同的label，就不切分了
	if len(y5.unique())==1:
		x5=x6=x4
		y5=y6=y4
	'''
	
	max=0
	
	# 外层（每行）是C，内层（每列）是gamma
	for ii in list1:
		for jj in list2:
			svc=SVC(C=ii,gamma=jj,cache_size=8000,decision_function_shape='ovo')
			svc.fit(x5,y5)				
			score.loc[ii,jj]=svc.score(x6,y6)
			print('------------------------------------------------------------------------------')
			time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
			print('mall_id='+i+',,,,'+time_now+',,,,'+str(len(x5)))
			#记录最大值
			print(score)
			if max<score.loc[ii,jj]:
				max=score.loc[ii,jj]
				ii_max=ii
				jj_max=jj
			#gc.collect()
	
	#重新训练
	svc=SVC(C=ii_max,gamma=jj_max,cache_size=8000,decision_function_shape='ovo')
	svc.fit(x5,y5)				
	print('mall_id='+i+',,,,'+time_now+',,,,'+str(len(x5)))
	way_tempt_svc='C:/Users/Administrator/Desktop/ali/data/3_tempt/svc/svc_'+i+'.model'
	joblib.dump(svc,way_tempt_svc)
	print('********************************************************************************')
	print('********************************************************************************')
	print('********************************************************************************')
	#gc.collect()
	
	
	






















