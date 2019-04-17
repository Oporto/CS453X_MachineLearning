import numpy as np
import matplotlib.pyplot as plt

def color(labels):
    label = np.argmax(labels)+1
    return '#'+hex(0x199999 * label)[2:]

def PCA(X, y):
    X_mean = np.mean(X, axis = 1).reshape(-1,1)
    X_s = X - X_mean
    X_s2 = X_s.transpose().dot(X_s)
    vals, vects = np.linalg.eig(X_s2)
    sorted_indices = np.argsort(vals)
    top_vs = vects[sorted_indices[-2:]]
    projection = top_vs.dot(X.transpose()).transpose()

    for i, point in enumerate(projection):
        plt.plot(point[0], point[1], color=color(y[i]), marker="o")
    plt.show()
    
if __name__ == "__main__":
    test_images = np.load("small_mnist_test_images.npy").reshape(-1, 784)
    test_values = np.load("small_mnist_test_labels.npy")
    PCA(test_images, test_values)