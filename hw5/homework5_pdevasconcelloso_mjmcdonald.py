import numpy as np
import matplotlib.pyplot as plt

def color(labels):
    label = np.argmax(labels)+1
    return '#'+hex(0x199999 * label)[2:]

def PCA(X, y):
    X_mean = np.mean(X, axis = 0).reshape(-1,1)
    X_2 = np.zeros((784,784))
    for i in range(X.shape[0]):
        X_2 += (X[i,:].reshape(784,1) - X_mean).dot((X[i,:].reshape(784,1) - X_mean).T)
    vals, vects = np.linalg.eig(X_2)
    sorted_indices = np.argsort(vals)
    top_vs = vects[:,sorted_indices[-2:]]
    projection = top_vs.T.dot(X.transpose()).transpose()

    for i, point in enumerate(projection):
        plt.plot(point[0], point[1], color=color(y[i]), marker="o")
    plt.show()
    
if __name__ == "__main__":
    test_images = np.load("small_mnist_test_images.npy").reshape(-1, 784)
    test_values = np.load("small_mnist_test_labels.npy")
    PCA(test_images, test_values)