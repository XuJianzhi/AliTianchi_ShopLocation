

#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import sklearn as sk
#import matplotlib.pyplot as plt  


way_write='C:/Users/Administrator/Desktop/ali/data/3_tempt/'
x=pd.read_csv(way_write+'x.csv',index_col=0)
y=pd.read_csv(way_write+'y.csv',index_col=0,header=None)	#否则会把第一行作为列表
#z=pd.read_csv(way_write+'z.csv',index_col=0,header=None)

'''
way4='C:/Users/Administrator/Desktop/ali/data/3_tempt/fenlei1.csv'
fenlei1=pd.read_csv(way4,header=None)
'''

#-----------------------------------------------------
#为两部kmeans而修改

xx,yy=x,y
from sklearn.cluster import KMeans
import gc
from sklearn.preprocessing import StandardScaler


#svc的y如果有其中一类只有一个样本，就会error，所有要去掉这个
def drop_lines(x,y):
	to_drop=(y.groupby(y).size()==1)
	to_drop=pd.Series(to_drop[to_drop].index)
	xx=x.copy()
	yy=y.copy()
	j=0
	for i in xrange(len(to_drop)):
		kkk= yy!=to_drop[i]
		xx=xx[kkk]
		yy=yy[kkk]
		j+=1
		#gc.collect() 	# 及时回收内存,要不就爆了
	return xx,yy

from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import time

#保存这种情况的i和j，这种情况是“所有样本有一个共同的label”
just_one_label=pd.DataFrame(columns=['i','j','label'])

for i in range(2,25):
	#print('			i='+str(i))
	way_fenlei_tempt='C:/Users/Administrator/Desktop/ali/data/3_tempt/second_kmeans/fenlei2_'+str(i)+'.csv'
	fenlei_tempt=pd.read_csv(way_fenlei_tempt,index_col=0,header=None)
	#for j in fenlei_tempt[1].unique():		#由于此行出来的顺序是乱的，一error就废了，所以改成有序的
	for j in range(40):
		print('i='+str(i)+',	j='+str(j))
		xuanze=((fenlei_tempt==j)[1])
		
		#报错原因之一：svc的y如果有其中一类只有一个样本，就会error，所有要去掉这个
		x3,y3=drop_lines(xx.loc[xuanze[xuanze].index,:],yy.loc[xuanze[xuanze].index,:][1])
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
			print('just_one_label')
			#print(just_one_label)
			
			#保存只有一种label的i和j
			way_tempt_2='C:/Users/Administrator/Desktop/ali/data/3_tempt/svc/same_label.scv'
			just_one_label.to_csv(way_tempt_2)
			print('------------------------------------------------------------------------------')
			time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
			print('i='+str(i)+',j='+str(j)+',,,,'+time_now+',,,,'+'situation 3')
			continue
		
		#标准化及其保存
		scale=StandardScaler()
		x4=scale.fit_transform(x3)
		y4=y3
		way_tempt_scale='C:/Users/Administrator/Desktop/ali/data/3_tempt/svc/scale_i'+str(i)+'_j'+str(j)+'.model'
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
		
		#防止切分后的训练集y5全是相同的label，就不切分了
		if len(y5.unique())==1:
			x5=x6=x4
			y5=y6=y4
		
		
		max=0
		
		# 外层（每行）是C，内层（每列）是gamma
		for ii in list1:
			for jj in list2:
				svc=SVC(C=ii,gamma=jj,cache_size=8000,decision_function_shape='ovo')
				svc.fit(x5,y5)				
				score.loc[ii,jj]=svc.score(x6,y6)
				print('------------------------------------------------------------------------------')
				time_now=time.strftime('%H:%M:%S',time.localtime(time.time()))
				print('i='+str(i)+',j='+str(j)+',,,,'+time_now+',,,,'+str(len(x5)))
				#记录最大值
				print(score)
				if max<score.loc[ii,jj]:
					max=score.loc[ii,jj]
					ii_max=ii
					jj_max=jj
				
				
				#gc.collect()
		#print(score)	#22:07
		
		#重新训练
		svc=SVC(C=ii_max,gamma=jj_max,cache_size=8000,decision_function_shape='ovo')
		svc.fit(x5,y5)				
		print('**********ii='+str(ii_max)+'****jj='+str(jj_max)+'****n='+str(len(x5)))
		way_tempt_svc='C:/Users/Administrator/Desktop/ali/data/3_tempt/svc/params2_i'+str(i)+'_j'+str(j)+'.model'
		#params_tempt=pd.Series(svc.get_params())
		#params_tempt.to_csv(way_tempt)
		
		joblib.dump(svc,way_tempt)
		

		print('--------------------------')
		#gc.collect()
		
		
		
		
		'''
i0,j2	未切分：
          0.01      0.10      1.00      10.00     100.00    1000.00
0.01     0.339941  0.339941  0.366590  0.415263  0.380134  0.349631
0.10     0.339941  0.385530  0.438388  0.462945  0.503249  0.474507
1.00     0.386631  0.434424  0.450171  0.505891  0.532430  0.563484
10.00    0.431891  0.448739  0.465147  0.520207  0.548508  0.601806
100.00   0.444004  0.455016  0.484969  0.530228  0.568109  0.634622
1000.00  0.450831  0.463826  0.508424  0.539478  0.585398  0.664464		
		
		已切分，行列简单：
          0.001     0.100     10.000    1000.000
0.001     0.352229  0.337369  0.342323  0.335168
0.100     0.326912  0.365988  0.459549  0.442488
10.000    0.369840  0.444689  0.503577  0.515685
1000.000  0.444689  0.451293  0.542653  0.495872	
		
		已切分，更详细：
          0.01      0.10      1.00      10.00     100.00    1000.00
0.01     0.342873  0.350578  0.362135  0.420473  0.368740  0.327463
0.10     0.353880  0.376445  0.434232  0.464502  0.495322  0.463401
1.00     0.393506  0.424876  0.456797  0.501926  0.510182  0.512933
10.00    0.417722  0.454045  0.470006  0.519538  0.527793  0.506879
100.00   0.455696  0.459549  0.464502  0.498624  0.515135  0.497523
1000.00  0.438085  0.445790  0.490919  0.531095  0.504128  0.478261		
		
		已切分，局部详细：
          1         10        100       1000      10000
1      0.440286  0.480462  0.496423  0.495872  0.465052
10     0.449092  0.490919  0.500275  0.485416  0.455696
100    0.462300  0.497523  0.494221  0.473858  0.440837
1000   0.478811  0.495322  0.488718  0.463952  0.428729
10000  0.487617  0.491469  0.488167  0.468905  0.418822
		更局部：
          20        40        60        80        100
20   0.518437  0.514034  0.512933  0.514034  0.513484
40   0.517336  0.515135  0.510182  0.512383  0.515685
60   0.514034  0.509081  0.511282  0.513484  0.514034
80   0.515685  0.511282  0.509081  0.514584  0.515685
100  0.512933  0.508531  0.508531  0.513484  0.513484
		
		
*************************************
*************************************
i,j=5,4

直接sgd的结果是0.358472998138

          0.01      0.10      1.00      10.00     100.00    1000.00
0.01     0.144320  0.144320  0.352886  0.468343  0.365922  0.268156
0.10     0.145251  0.361266  0.513035  0.603352  0.644320  0.535382
1.00     0.366853  0.517691  0.594972  0.661080  0.707635  0.690875
10.00    0.517691  0.581937  0.638734  0.687151  0.708566  0.689944
100.00   0.568901  0.595903  0.659218  0.702980  0.702048  0.671322
1000.00  0.591248  0.624767  0.673184  0.708566  0.694600  0.662011		
		
100     0.706704  0.716015
1000    0.710428  0.710428
10000   0.710428  0.696462
100000  0.715084  0.689944	
		
		
		
		
		
		
		
		'''
	
	
	
'''	
#######
	xuanze=((fenlei1==i)[0])
	#svc的y如果有其中一类只有一个样本，就会error，所有要去掉这个
	#x3,y3=drop_lines(xx[xuanze],yy[xuanze])
	x3,y3=xx[xuanze],yy[xuanze]
		
	#实验中，当i=21时仅有一个样本。经drop_lines后为空集。
	#if len(x3)==0:	continue
	
	svc=SVC(C=1,cache_size=2000,decision_function_shape='ovo')
	svc.fit(x3,y3)
	print('++++++++++++++++')
	print(svc.score(xx[xuanze],yy[xuanze]))
	way_tempt='C:/Users/Administrator/Desktop/ali/data/3_tempt/params2'+str(i)+'.csv'
	params_tempt=pd.Series(svc.get_params())
	params_tempt.to_csv(way_tempt)
	print('--------------------------')
	#gc.collect()
'''

# 第三版改动为将index也写入csv
# 第四版使用joblib将模型持久化
# 第五版在svc前加入scale





















