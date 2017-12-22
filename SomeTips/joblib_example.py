from sklearn.externals import joblib
#lr是一个LogisticRegression模型
joblib.dump(lr, 'lr.model')
lr = joblib.load('lr.model')