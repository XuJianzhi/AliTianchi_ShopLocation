


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
way1='/home/m/桌面/2017.10.24/ali/data/1_use/1_ccf_first_round_shop_info.csv'
way2='/home/m/桌面/2017.10.24/ali/data/1_use/2_ccf_first_round_user_shop_behavior.csv'
way3='/home/m/桌面/2017.10.24/ali/data/1_use/3_evaluation_public.csv'

data1=pd.read_csv(way1)
data2=pd.read_csv(way2)
data3=pd.read_csv(way3)

data=pd.merge(data1,data2,on='shop_id',suffixes=('_real','_variable'))



x=data[['longitude_variable','latitude_variable']]
y=data['shop_id']



#*******************svm太差了，放弃了*****************
#*****************************************************
#*****************************************************
#******************尝试神经网络*************************
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation  
from keras.optimizers import SGD  
from keras.datasets import mnist  
import numpy 

'''
    第一步：选择模型
'''
model = Sequential()
'''
   第二步：构建网络层
'''
model.add(Dense(1000,input_shape=(2,))) # 输入层，28*28=784  
model.add(Activation('tanh')) # 激活函数是tanh  
model.add(Dropout(0.5)) # 采用50%的dropout

model.add(Dense(500)) # 隐藏层节点500个  
model.add(Activation('tanh'))  
model.add(Dropout(0.5))

model.add(Dense(10)) # 输出结果是10个类别，所以维度是10  
model.add(Activation('softmax')) # 最后一层用softmax作为激活函数

'''
   第三步：编译
'''
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # 优化函数，设定学习率（lr）等参数  
model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical') # 使用交叉熵作为loss函数

'''
   第四步：训练
   .fit的一些参数
   batch_size：对总的样本数进行分组，每组包含的样本数量
   epochs ：训练次数
   shuffle：是否把数据随机打乱之后再进行训练
   validation_split：拿出百分之多少用来做交叉验证
   verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果
'''

n=100000
xx,yy=x1[:n],y[:n]


#(X_train, y_train), (X_test, y_test) = mnist.load_data() # 使用Keras自带的mnist工具读取数据（第一次需要联网）
# 由于mist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维  
#X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]) 
#X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])  
Y_train = (numpy.arange(10) == y_train[:, None]).astype(int) 
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

model.fit(X_train,Y_train,batch_size=200,epochs=50,shuffle=True,verbose=0,validation_split=0.3)
model.evaluate(X_test, Y_test, batch_size=200, verbose=0)








