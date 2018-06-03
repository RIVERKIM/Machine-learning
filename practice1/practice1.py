from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC

import pickle
from sklearn.externals import joblib
import numpy as np
from sklearn import random_projection
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer


iris=datasets.load_iris()
digits=datasets.load_digits()
print(digits.data)

print(digits.target)
print(digits.images[0])

clf=svm.SVC(gamma=0.001,C=100.)
print(clf.fit(digits.data[:-1],digits.target[:-1]))
print(clf.predict(digits.data[-1:]))

clf=svm.SVC()
iris=datasets.load_iris()
X,y=iris.data,iris.target
print(clf.fit(X,y))

s=pickle.dumps(clf)
clf2=pickle.loads(s)
print(clf2.predict(X[0:1]))

print(joblib.dump(clf,'filename.pkl'))
print(clf=joblib.load('filename.pkl'))

rng=np.random.RandomState(0)
X=rng.rand(10,2000)
X=np.array(X,dtype='float32')
print(X.dtype)

transformer=random_projection.GaussianRandomProjection()
X_new=transformer.fit_transform(X)
print(X_new.dtype)

iris=datasets.load_iris()
clf=SVC()
print(clf.fit(iris.data,iris.target))

print(list(clf.predict(iris.data[:3])))
print(clf.fit(iris.data,iris.target_names[iris.target]))

print(list(clf.predict(iris.data[:3])))

rng=np.random.RandomState(0)
X=rng.rand(100,10)
y=rng.binomial(1,0.5,100)
X_test=rng.rand(5,10)

clf=SVC()
print(clf.set_params(kernel='linear').fit(X,y))

print(clf.predict(X_test))

print(clf.set_params(kernel='rbf').fit(X,y))

print(clf.predict(X_test))

X=[[1,2],[2,4],[4,5],[3,2],[3,1]]
y=[0,0,1,1,2]

classif=OneVsRestClassifier(estimator=SVC(random_state=0))
print(classif.fit(X,y).predict(X))

y=LabelBinarizer().fit_transform(y)
print(classif.fit(X,y).predict(X))

y=[[0,1],[0,2],[1,3],[0,2,3],[2,4]]
y=MultiLabelBinarizer().fit_transform(y)
print(classif.fit(X,y).predict(X))