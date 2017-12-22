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





location_source=data[['shop_id','longitude_real','latitude_real','longitude_variable','latitude_variable']]
location_source=location_source.set_index(['shop_id'])

#将6为小数都换成一位小数
location=location_source.copy()
location=location*100000

location=location.reset_index()

'''
############### 求每种店铺的平均、上下左右 #############

detail=pd.DataFrame()
group=location.groupby(level=['shop_id'])
detail['longitude_min']=group['longitude_variable'].min()
detail['longitude_max']=group['longitude_variable'].max()
detail['longitude_real']=group['longitude_real'].max()

detail['latitude_min']=group['latitude_variable'].min()
detail['latitude_max']=group['latitude_variable'].max()
detail['latitude_real']=group['latitude_real'].max()
'''

######## 求有中心点；在每一个店铺里求每个点与最近的一个点的距离，第一名与第二名差太多就剔除；得到半径	######

location=data[['shop_id','longitude_real','latitude_real','longitude_variable','latitude_variable']]
#location=location.set_index(['shop_id'])
shop=location.groupby(location['shop_id']).size()
location['distance']=np.nan

for s in shop.index:
	part=location[location['shop_id']==s]
	for i in part.index:
		part['distance_temp']=(part['longitude_variable']-part['longitude_variable'][i])**2\
			+(part['latitude_variable']-part['latitude_variable'][i])**2
	for j in 
			
			
			
#s=shop.index[2];i=part.index[1];part=location[location['shop_id']==s]













