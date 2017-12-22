

#!/usr/bin/python
# -*- coding: UTF-8 -*-

#第一步只考虑经纬度影响：先按mall分类，每个mall里对经纬度取mean
#第二步分析wifi信号对店铺的影响
#第三步分析时间（是否放假、几点）对店铺的影响，出图说明


########################	求店铺半径，在knn时加权值	################


import pandas as pd
import numpy as np
import sklearn as sk
#import matplotlib.pyplot as plt  
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


# 提取
way1='C:/Users/Administrator/Desktop/ali/data/1_use/1_ccf_first_round_shop_info.csv'
way2='C:/Users/Administrator/Desktop/ali/data/1_use/2_ccf_first_round_user_shop_behavior.csv'
#way3='C:/Users/Administrator/Desktop/ali/data/1_use/3_evaluation_public.csv'

data1=pd.read_csv(way1)
data2=pd.read_csv(way2)
#data3=pd.read_csv(way3)

data=pd.merge(data1,data2,on='shop_id',suffixes=('_real','_variable'))

x=data[['longitude_variable','latitude_variable']]

#标准化
scale=StandardScaler()
xx=pd.DataFrame(scale.fit_transform(x))

data['longitude_variable']=xx[0]
data['latitude_variable']=xx[1]

train=data[['longitude_variable','latitude_variable','mall_id','category_id','shop_id']]
mall_list=pd.Series(train['mall_id'].unique())
category_list=pd.Series(train['category_id'].unique())
wifi=data['wifi_infos']

#储存数据
way_write='C:/Users/Administrator/Desktop/ali/data/4_knn/'

train.to_csv(way_write+'train.csv',index=False)
mall_list.to_csv(way_write+'mall_list.csv',index=False)
category_list.to_csv(way_write+'category_list.csv',index=False)
wifi.to_csv(way_write+'wifi.csv',index=False)

joblib.dump(scale, way_write+'scale.model')




