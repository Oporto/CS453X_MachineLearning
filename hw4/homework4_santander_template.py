import sklearn.svm
import sklearn.metrics
import numpy as np
import pandas

# Load data
df = pandas.read_csv('train.csv')
y = np.array(df.target)  # Labels
X = np.array(df.iloc[:,2:])  # Features


def compute_one_go(trainx, trainy, testx):
    # Split into train/test folds
    
    
    # Linear SVM
    
    svm_lin = sklearn.svm.SVC(kernel='linear', C=1e15)
    svm_lin.fit(train_x, train_y)
    
    # Non-linear SVM (polynomial kernel)
    
    svm_poly = sklearn.svm.SVC(kernel='poly', degree=3, C=1e15)
    svm_poly.fit(train_x, train_y)
    
    # Apply the SVMs to the test set
    yhat_lin = svm_lin.predict(test_x)
    yhat_poly = svm_poly.predict(test_x)
    return yhat_lin, yhat_poly
    
    
def compute_bag(trainx, trainy, testx, testy):
    indexes = np.arange(trainy.shape[0])
    np.random.shuffle(indexes)
    indexes.reshape(5,-1)
    pred_lin = np.zeros(trainy.shape[0])
    pred_poly = np.zeros(trainy.shape[0])
    
    for i in range(5):
        ind = indexes[i]
        lin, poly = compute_one_go(trainx[ind], trainy[ind], testx)
        pred_lin.add(lin)
        pred_poly.add(poly)
    
    yhat_lin = pred_lin/5
    yhat_poly = pred_poly/5
    auc_lin = sklearn.metrics.roc_auc(test_y, yhat_lin)
    auc_poly = sklearn.metrics.roc_auc(test_y, yhat_poly)
    return auc_lin, auc_poly
    

if __name__ == "__main__": 
    sample_size = len(df)
    print(sample_size)
    train_i = np.random.rand(len(df)) < 0.5
    train_x = X[train_i]
    test_x = X[~train_i]
    train_y = y[train_i]
    test_y = y[~train_i]
    
    auc_lin, auc_poly = compute_one_go(trainx, trainy, testx)
    # Compute AUC
    auc_lin = sklearn.metrics.roc_auc(test_y, yhat_lin)
    auc_poly = sklearn.metrics.roc_auc(test_y, yhat_poly)
    print("No bag:")
    print(auc_lin)
    print(auc_poly)
    
    bag_auc_lin, bag_auc_poly = compute_bag(trainx, trainy, testx, testy)
    print("In 5 bags:")
    print(bag_auc_lin)
    print(bag_auc_poly)

