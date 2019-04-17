import numpy as np

def PCA(X, y):
    X_mean = np.mean(X, axis = 1).reshape(-1,1)
    X_s = X - X_mean
    X_s2 = np.apply_along_axis(lambda x: x.reshape(28,28).dot(x.reshape(28,28).transpose()), 1, X_s).shape
    vals, vects = np.linalg.eig(X_s2)
    top_indices = np.argsort(vals,1)
    X_vs = []
    for i in range(5000):
        X_vs.append(vects[i][top_indices[i][-2:]])
    X_vs = np.array(X_vs)
    
    projections = np.tensordot(X.reshape(-1,28,28), X_vs, axes=(1,2))
    
if __name__ == "__main__":
    test_images = np.load("small_mnist_test_images.npy").reshape(-1, 784)
    test_values = np.load("small_mnist_test_labels.npy")