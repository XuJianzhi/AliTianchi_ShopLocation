
'''
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


# 提取
way1='C:/Users/Administrator/Desktop/ali/data/1_use/1_ccf_first_round_shop_info.csv'
way2='C:/Users/Administrator/Desktop/ali/data/1_use/2_ccf_first_round_user_shop_behavior.csv'
way3='C:/Users/Administrator/Desktop/ali/data/1_use/3_evaluation_public.csv'

data1=pd.read_csv(way1)
data2=pd.read_csv(way2)
data3=pd.read_csv(way3)

data=pd.merge(data1,data2,on='shop_id',suffixes=('_real','_variable'))



x=data[['longitude_variable','latitude_variable']]
y=data['shop_id']
z=data['mall_id']

way_write='C:/Users/Administrator/Desktop/ali/data/3_tempt/'

x.to_csv(way_write+'x.csv')
y.to_csv(way_write+'y.csv')
z.to_csv(way_write+'z.csv')
'''


#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import sklearn as sk
#import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split

way_write='C:/Users/Administrator/Desktop/ali/data/3_tempt/'
x=pd.read_csv(way_write+'x.csv',index_col=0)	#以列标叫0的列作为index
y=pd.read_csv(way_write+'y.csv',index_col=0,header=None)[1]		#否则会把第一行作为列表
#z=pd.read_csv(way_write+'z.csv',index_col=0,header=None)

#------------------------------------------------------------------------
from sklearn.linear_model import SGDClassifier

nn=200000
xx=x[:nn];yy=y[:nn]
x5,x6,y5,y6=train_test_split(xx, yy, test_size=0.2)

print('-----')
clf=SGDClassifier(loss='log',max_iter=10)
print('-----++++++')
clf.fit(x5,y5)
#clf.partial_fit(x,y,classes=y.unique())
print('-----++++++*******')
print(clf.score(x6,y6))

'''
在nn=200000时，loss为默认，	结果时0.018125
在nn=200000时，loss为log，	结果时0.0798
都太惨啦
'''








