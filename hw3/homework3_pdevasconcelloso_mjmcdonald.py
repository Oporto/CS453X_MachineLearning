import numpy as np
import matplotlib.pyplot as plt
import math
import sys


def find_cost(predict, actual):
    diff_squared = np.square(predict - actual)
    return np.average(diff_squared) / 2


def find_cost_with_penalty(predict, actual, weight, alpha):
    diff_squared = np.square(predict - actual)
    old_cost = np.average(diff_squared) / 2
    penalty_term = (alpha / (2*len(actual)))*weight*np.transpose(weight)
    return np.average(old_cost + penalty_term)

def find_rmse(predict, actual):
    return np.sqrt(np.square(predict - actual)).mean()
    
def calc_prediction(X, aug_w):
    w = aug_w[:-1]
    b = aug_w[-1]
    pred = X.dot(w) + b
    return pred

    
def gradient(X, y, w, b):
    y_hat = calc_prediction(augment(X), np.hstack((w, b)))
    gradient = X * (y_hat - y) / X.shape[0]
    delta = np.average(gradient)
    return gradient, delta


def predicts(train_images, test_images, train_values, test_values, aug_w, name):
    train_pred = calc_prediction(train_images, aug_w)
    test_pred = calc_prediction(test_images, aug_w)
    train_cost = find_cost(train_pred, train_values)
    test_cost = find_cost(test_pred, test_values)
    np.save(name, aug_w)
    print(name+": \n")
    print("Train cost: ", train_cost)
    print("Test cost: ", test_cost)

    im = aug_w[0:-1]
    im = im.reshape(48, 48)
    plt.imshow(im, cmap='gray')
    plt.title(name)
    plt.show()
    return train_pred, test_pred
    

def augment(X):
    return np.hstack((X, np.transpose(np.atleast_2d(np.ones(X.shape[0])))))


def stochastic_gradient_descent(epochs, batch_size, epsilon, train_images, test_images, train_values, test_values):
    aug_w = generate_weights()
    w = aug_w[:-1]
    b = aug_w[-1]
    N = train_images.shape[0]

    train_fused = np.hstack((train_images, train_values))
    np.random.shuffle(train_fused)
    train_images = train_fused[:,:-10]
    train_values = train_fused[:,-10:]

    for epoch in range(epochs):
        for round in range(math.ceil(N / batch_size)):
            sample_img = train_images[round*batch_size:(round+1)*batch_size]
            sample_val = train_values[round*batch_size:(round+1)*batch_size]

            g, delta = gradient(sample_img, sample_val, w, b)
            w = w - epsilon*g
            b = b - epsilon*delta

    aug_w = np.hstack((w, b))
    return predicts(train_images, test_images, train_values, test_values, aug_w)


def generate_weights():
    sigma = 0.01 ** 0.5
    mu = 0.5
    return sigma * np.random.randn(24*24+1) + mu


if __name__ == "__main__":
    train_images = np.load("small_mnist_train_images.npy").reshape(-1, 784)
    test_images = np.load("small_mnist_test_images.npy").reshape(-1, 784)
    train_values = np.load("small_mnist_train_labels.npy")
    test_values  = np.load("small_mnist_test_labels.npy")
    m3train, m3test = stochastic_gradient_descent(100, 100, 0.1, train_images, test_images, train_values, test_values)
