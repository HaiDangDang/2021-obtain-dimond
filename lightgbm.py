# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
import pickle
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import treelite
import treelite_runtime
from hummingbird.ml import convert
import xgboost as xgb
import time

data_train = {}
with open('data_train.pkl', 'rb') as f:
    data_train = pickle.load(f)

X = data_train['x']
y = data_train['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 10)
clf = lgb.LGBMClassifier(device='gpu', boosting_type='gbdt', num_leaves=80, max_depth=7, learning_rate=0.01, n_estimators=80,
                         subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001,
                         min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0,
                         reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=1, silent='warn', importance_type='split')
clf.fit(X_train, y_train)


tmp_time = time.time()
y_pred=clf.predict(X_test)

print(time.time() - tmp_time)

print(accuracy_score(y_pred, y_test))

with open('first_model.pkl', 'wb') as fout:
    pickle.dump(clf, fout)


tmp_time = time.time()
dmat = treelite_runtime.DMatrix(X)
out_pred = predictor.predict(dmat)
print(time.time() - tmp_time)
with open('first_model.pkl', 'rb') as fin:
    clf = pickle.load(fin)

accuracy=accuracy_score(y_pred, y_test)


le = LabelEncoder()
data_train = {}
with open('data_train.pkl', 'rb') as f:
    data_train = pickle.load(f)

X = data_train['x']
y = data_train['y']
y = le.fit_transform(y)
le.classes_

y_train = le.fit_transform(y_train)
d_train=lgb.Dataset(X_train, label=y_train)

#Specifying the parameter
params={}
params['learning_rate']=0.01
params['boosting_type']='dart' #GradientBoostingDecisionTree
params['objective']='multiclass' #Multi-class target feature
params['metric'] ='multi_logloss' #metric for multi-class
params['device'] ='gpu'
params['num_leaves']=80
params['max_depth']=7
params['subsample_for_bin']=200000
params['n_jobs'] = 3
params['importance_type']= 'split'
params['n_estimators'] = 80
params['num_class'] = len(le.classes_)
np.save('name_classes_.npy',le.classes_)
clf = lgb.train(params,d_train) #train the model on 100 epocs

with open('second_model.pkl', 'wb') as fout:
    pickle.dump(clf, fout)
y_pred = clf.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
y_pred.shape

u = np.unique(y)
len(u)
y.shape
for i in u:
    print(i, np.sum(y == i))

lgb.plot_importance(clf)
plt.show()
X[500,1]
for i in range(X.shape[0]):
    print(np.mean(X[:,i]))
a = X[:,8]
np.max(a)
np.histogram(a)
