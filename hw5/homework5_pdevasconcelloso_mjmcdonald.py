import numpy as np

def PCA(X, y):
    X_mean = np.mean(X, axis = 0).reshape(-1,1)
    X_2 = np.zeros((784,784))
    for i in range(X.shape[0]):
        X_2 += (X[i,:].reshape(784,1) - X_mean).dot((X[i,:].reshape(784,1) - X_mean).T)

    print(X_2.shape)
    vals, vects = np.linalg.eig(X_2)
    print(vals.shape)
    sorted_indices = np.argsort(vals)
    top_vs = vects[sorted_indices[-2:]]
    projection = top_vs.dot(X.transpose()).transpose()
    
if __name__ == "__main__":
    test_images = np.load("small_mnist_test_images.npy").reshape(-1, 784)
    test_values = np.load("small_mnist_test_labels.npy")
    PCA(test_images[:100], test_values[:100])