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

#way1='C:/Users/Administrator/Desktop/ali/data/1_use/1_ccf_first_round_shop_info.csv'
#way2='C:/Users/Administrator/Desktop/ali/data/1_use/2_ccf_first_round_user_shop_behavior.csv'
way3='C:/Users/Administrator/Desktop/ali/data/1_use/3_evaluation_public.csv'

#data1=pd.read_csv(way1)
#data2=pd.read_csv(way2)
data3=pd.read_csv(way3)

#恢复文件一的训练
way_write='C:/Users/Administrator/Desktop/ali/data/3_tempt/'
scale = joblib.load(way_write+'scale_1.model')
test=pd.DataFrame({0:data3['longitude'],1:data3['latitude']})
test_scaled=pd.DataFrame(scale.transform(test))
way_write='C:/Users/Administrator/Desktop/ali/data/3_tempt/'	#保存test_scaled
test_scaled.to_csv(way_write+'test_scaled.csv')

#恢复文件二中第一层的训练
way5='C:/Users/Administrator/Desktop/ali/data/3_tempt/kmeans_1.model'
kmeans=joblib.load(way5)
fenlei1_test=kmeans.predict(test_scaled)


#恢复文件二中第二层的训练
test_list=pd.Series(np.arange(25))
for i in test_list:
	print(i)
	xuanze=((fenlei1_test==i))
	
	if len(test_scaled[xuanze])==0:
		test_list=test_list.drop(i)
		print('no one')
		continue

	way_params_tempt='C:/Users/Administrator/Desktop/ali/data/3_tempt/second_kmeans/kmeans_2_'+str(i)+'.model'
	kmeans_tempt_test=joblib.load(way_params_tempt)
	fenlei_tempt_test=pd.Series(kmeans_tempt_test.predict(test_scaled[xuanze]),index=test_scaled[xuanze].index)	#predict后index清零，故需手动配上
	#存储分类结果
	way_fenlei_tempt_test='C:/Users/Administrator/Desktop/ali/data/3_tempt/second_kmeans_test/fenlei2_'+str(i)+'.csv'
	fenlei_tempt_test.to_csv(way_fenlei_tempt_test)


#-----------------------------------------------------------------
#读取基本数据test_scaled
way_write='C:/Users/Administrator/Desktop/ali/data/3_tempt/'
test_scaled=pd.read_csv(way_write+'test_scaled.csv',index_col=0)

#恢复文件三的训练
way_tempt_2='C:/Users/Administrator/Desktop/ali/data/3_tempt/svc/same_label.scv'	#########
just_one_label=pd.read_csv(way_tempt_2,index_col=0)		#仅有一种label的i，j情况
result=pd.Series()	#储存结果
for i in test_list:
	print('*************i='+str(i))
	way_fenlei_tempt_test='C:/Users/Administrator/Desktop/ali/data/3_tempt/second_kmeans_test/fenlei2_'+str(i)+'.csv'
	fenlei_tempt_test=pd.read_csv(way_fenlei_tempt_test,index_col=0,header=None)
	#for j in fenlei_tempt[1].unique():		#由于此行出来的顺序是乱的，一error就废了，所以改成有序的
	for j in range(40):
		print('------------------------------------------------------------------------------')
		print('i='+str(i)+',	j='+str(j))
		xuanze=((fenlei_tempt_test==j)[1])
		#读取数据
		x3=test_scaled.loc[xuanze[xuanze].index,:]
		
		#用于检测空集
		if len(x3)==0:	
			print('00000000000000000000000000000000')
			continue
			
		#用于检测在just one label表格中的情况，直接出答案
		if (i in np.array(just_one_label['i'])) and (j in np.array(just_one_label[just_one_label['i']==i]['j'])):
			label=np.array(just_one_label[just_one_label['i']==i][just_one_label['j']==j]['label'])
			y3=pd.Series(label*len(x3),index=x3.index)
			#储存结果
			result=pd.concat([result,y3])
			time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
			print('i='+str(i)+',j='+str(j)+',,,,'+time_now+',,,,'+str(len(x3)))
			continue
		
		#恢复svc训练
		way_tempt='C:/Users/Administrator/Desktop/ali/data/3_tempt/svc/params2_i'+str(i)+'_j'+str(j)+'.model'		#############
		svc=joblib.load(way_tempt)
		#恢复scale的训练
		way_tempt_scale='C:/Users/Administrator/Desktop/ali/data/3_tempt/svc/scale_i'+str(i)+'_j'+str(j)+'.model'
		scale=joblib.load(way_tempt_scale)
		x4=pd.DataFrame(scale.fit_transform(x3),index=x3.index)
				
		#开始预测
		y4=pd.Series(svc.predict(x4),index=x4.index)
		#储存结果
		result=pd.concat([result,y4])
		
		time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
		print('i='+str(i)+',j='+str(j)+',,,,'+time_now+',,,,'+str(len(x4)))

		
way_result='C:/Users/Administrator/Desktop/ali/data/3_tempt/result.csv'
result.to_csv(way_result)












