from sklearn.calibration import CalibratedClassifierCV 
import numpy as np
import pandas as pd

a=pd.DataFrame(np.arange(9).reshape(3,3))
#b=pd.Series(list('xyz'))
b=pd.Series(np.arange(3))


cv=CalibratedClassifierCV()
cv.fit(a,b)
print(cv.get_params)
print(predict(a))
print(score(a,b))


#---------
# 无监督学习
# cluster可以用于n个分类，分为‘n_clusters’类
# 即使分类器只能分两类，也可用sklearn.multioutput.MultiOutputClassifier变为分多类
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0],
			  [7, 2], [7, 4], [7, 0]])
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print(kmeans.labels_)	#[2 2 2 1 1 1 0 0 0]
print(kmeans.predict([[0, 0], [4, 4],[7,2]]))	#[2 1 0]
print(kmeans.cluster_centers_)	#[[ 7.  2.],[ 4.  2.],[ 1.  2.]]
print(kmeans.get_params())
#{'n_jobs': 1, 'algorithm': 'auto', 'n_clusters': 3, 'max_iter': 300,
# 'init': 'k-means++', 'random_state': 0, 'n_init': 10, 'tol': 0.0001,
# 'precompute_distances': 'auto', 'copy_x': True, 'verbose': 0}
#--------------------------









