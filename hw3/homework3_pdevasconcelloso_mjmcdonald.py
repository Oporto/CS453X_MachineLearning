import numpy as np
import matplotlib.pyplot as plt
import math
import skimage.transform as skt


def find_cost(predict, actual):
    prod = np.multiply(actual, np.log(predict))
    k_summed = prod.sum(axis=1)
    return -np.average(k_summed)


def calc_prediction(aug_X, aug_w):
    z = np.exp(aug_X.dot(np.transpose(aug_w)))
    total = z.sum(axis=1)
    pred = np.transpose(z) / total.flatten()
    return np.transpose(pred)


def percent_correct(predict, actual):
    pred = np.argmax(predict, axis=1)
    act = np.argmax(actual, axis=1)
    bool_arr = np.equal(pred, act)
    return np.sum(bool_arr) / bool_arr.shape[0]


def gradient(X, y, w, b):
    aug_w = np.transpose(np.vstack((np.transpose(w), b)))
    y_hat = calc_prediction(augment(X), aug_w)
    gradient = np.transpose(X).dot((y_hat - y)) / X.shape[0]
    delta = np.mean(gradient, axis=0).flatten()
    return np.transpose(gradient), delta


def predicts(train_images, test_images, train_values, test_values, aug_w,
             name):
    train_pred = calc_prediction(augment(train_images), aug_w)
    test_pred = calc_prediction(augment(test_images), aug_w)
    train_cost = find_cost(train_pred, train_values)
    test_cost = find_cost(test_pred, test_values)
    np.save(name, aug_w)
    print(name + ": \n")
    print("Train cost: ", train_cost)
    print("Test cost: ", test_cost)
    print("Testing % correct", percent_correct(test_pred, test_values))

    return train_pred, test_pred


def augment(X):
    return np.hstack((X, np.transpose(np.atleast_2d(np.ones(X.shape[0])))))


def stochastic_gradient_descent(epochs, batch_size, epsilon, train_images,
                                test_images, train_values, test_values):
    aug_w = generate_weights()
    w = aug_w[:, :-1]
    b = aug_w[:, -1]
    N = train_images.shape[0]

    train_fused = np.hstack((train_images, train_values))
    np.random.shuffle(train_fused)
    train_images = train_fused[:, :-10]
    train_values = train_fused[:, -10:]

    for epoch in range(epochs):
        for round in range(math.ceil(N / batch_size)):
            sample_img = train_images[round * batch_size:(round + 1) *
                                      batch_size]
            sample_val = train_values[round * batch_size:(round + 1) *
                                      batch_size]

            g, delta = gradient(sample_img, sample_val, w, b)
            w = w - epsilon * g
            b = b - epsilon * delta

    aug_w = np.transpose(np.vstack((np.transpose(w), b)))
    return predicts(train_images, test_images, train_values, test_values,
                    aug_w, "SGD")


def generate_weights():
    sigma = 0.01**0.5
    mu = 0.5
    return (sigma * np.random.randn((28 * 28 + 1) * 10) + mu).reshape(
        10, 28 * 28 + 1)
        
def augment_rotation(X, labels):
    X1 = np.apply_along_axis(lambda x: skt.rotate(x.reshape(28,28), 90), 1, X)
    X2 = np.apply_along_axis(lambda x: skt.rotate(x.reshape(28,28), 180), 1, X)
    X3 = np.apply_along_axis(lambda x: skt.rotate(x.reshape(28,28), 270), 1, X)
    X_rots = np.vstack((X1,X2,X3))
    lab_rots = np.vstack((labels, labels, labels))
    return X_rots, lab_rots
    


if __name__ == "__main__":
    train_images = np.load("small_mnist_train_images.npy").reshape(-1, 784)
    test_images = np.load("small_mnist_test_images.npy").reshape(-1, 784)
    train_values = np.load("small_mnist_train_labels.npy")
    test_values = np.load("small_mnist_test_labels.npy")
    m3train, m3test = stochastic_gradient_descent(
        100, 100, 0.1, train_images, test_images, train_values, test_values)
