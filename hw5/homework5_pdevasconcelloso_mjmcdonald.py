import numpy as np

def PCA(X, y):
    X_mean = np.mean(X, axis = 1).reshape(-1,1)
    X_s = X - X_mean
    X_s2 = X_s.transpose().dot(X_s)
    vals, vects = np.linalg.eig(X_s2)
    sorted_indices = np.argsort(vals)
    top_vs = vects[sorted_indices[-2:]]
    projection = top_vs.dot(X.transpose()).transpose()
    
if __name__ == "__main__":
    test_images = np.load("small_mnist_test_images.npy").reshape(-1, 784)
    test_values = np.load("small_mnist_test_labels.npy")
    PCA(test_images[:100], test_values[:100])