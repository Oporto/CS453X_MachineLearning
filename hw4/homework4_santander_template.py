import sklearn.svm
import sklearn.metrics
import numpy as np
import pandas

# Load data
d = pandas.read_csv('train.csv')
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:,2:])  # Features

# Split into train/test folds
print(d.shape)
# TODO

# Linear SVM
# TODO
svm_lin = sklearn.svm.SVC(kernel='linear', C=1e15)
svm_lin.fit(X, y)

# Non-linear SVM (polynomial kernel)
# TODO
svm_poly = sklearn.svm.SVC(kernel='poly', degree=3, C=1e15)
svm_poly.fit(X, y)

# Apply the SVMs to the test set
#yhat1 = ...  # Linear kernel
#yhat2 = ...  # Non-linear kernel

# Compute AUC
#auc1 = ...
#auc2 = ...

# print(auc1)
# print(auc2)

