import pandas as pd
import numpy as np
import sklearn as sk
#import matplotlib.pyplot as plt  
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import gc

# 提取
way1='C:/Users/Administrator/Desktop/ali/data/1_use/1_ccf_first_round_shop_info.csv'
way2='C:/Users/Administrator/Desktop/ali/data/1_use/2_ccf_first_round_user_shop_behavior.csv'
way3='C:/Users/Administrator/Desktop/ali/data/1_use/3_evaluation_public.csv'

data1=pd.read_csv(way1)
data2=pd.read_csv(way2)
data3=pd.read_csv(way3)

data=pd.merge(data1,data2,on='shop_id',suffixes=('_real','_variable'))



#x=data[['longitude_variable','latitude_variable']]
#y=data['shop_id']
#z=data['mall_id']
w=data['wifi_infos']

#标准化
#scale=StandardScaler()
#xx=pd.DataFrame(scale.fit_transform(x))
df_wifi=pd.DataFrame()
for i in w:

	s1=i.split(';')
	s2=[np.nan]*len(s1)
	for j in range(len(s1)):
		s2[j]=s1[j].split('|')
	s3=pd.DataFrame(s2).set_index(0).drop(2,axis=1).T
	df_wifi=df_wifi.append(s3)
	print(df_wifi)
	gc.collect()











#储存数据
way_write='C:/Users/Administrator/Desktop/ali/data/3_tempt/'

xx.to_csv(way_write+'x.csv')
y.to_csv(way_write+'y.csv')
z.to_csv(way_write+'z.csv')
x.to_csv(way_write+'w.csv')

joblib.dump(scale, way_write+'scale_1.model')
