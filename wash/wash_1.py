#!/usr/bin/python
# -*- coding: UTF-8 -*-

#第一步只考虑经纬度影响：先按mall分类，每个mall里对经纬度取mean
#第二步分析wifi信号对店铺的影响
#第三步分析时间（是否放假、几点）对店铺的影响，出图说明

import pandas as pd
import numpy as np
import sklearn as sk
from collections import Counter

# 提取
way1='C:/Users/Administrator/Desktop/ali/data/1_use/1_ccf_first_round_shop_info.csv'
way2='C:/Users/Administrator/Desktop/ali/data/1_use/2_ccf_first_round_user_shop_behavior.csv'
way3='C:/Users/Administrator/Desktop/ali/data/1_use/3_evaluation_public.csv'

data1=pd.read_csv(way1)
data2=pd.read_csv(way2)
data3=pd.read_csv(way3)

data=pd.merge(data1,data2,on='shop_id',suffixes=('_real','_variable'))

changes=pd.DataFrame({'longitude':data['longitude_variable']-data['longitude_real'],\
	'latitude':data['latitude_variable']-data['latitude_real']})

# longitude_range=(max(data['longitude_real'])-min(data['longitude_real']))
# data['longitude_cooked']=(data['longitude_variable']-min(data['longitude_real']))/longitude_range

############### 求每种店铺的面积 #############

long_max=data.groupby(['mall_id','category_id'])[['longitude_real']].max()-\
	data.groupby(['mall_id','category_id'])[['longitude_real']].min()
width_max=data.groupby(['mall_id','category_id'])[['latitude_real']].max()-\
	data.groupby(['mall_id','category_id'])[['latitude_real']].min()
long_mean=data.groupby(['mall_id','category_id'])[['longitude_real']].mean()
width_mean=data.groupby(['mall_id','category_id'])[['latitude_real']].mean()


room=pd.DataFrame()
room['long_max']=long_max['longitude_real']
room['width_max']=width_max['latitude_real']
room['long_mean']=long_mean
room['width_mean']=width_max

rr=room.swaplevel().sortlevel()







