import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize
import math

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient


# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack(w):
    b2 = w[-1][:NUM_OUTPUT]
    w = w[:-1]
    W2 = w[-NUM_OUTPUT:].T
    w = w[:-NUM_OUTPUT]
    b1 = w[-1]
    W1 = w[:-1]
    return W1, b1, W2, b2


# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack(W1, b1, W2, b2):
    # 784x40, 40x1, 40x10, 10x1
    W1b1 = np.vstack((W1, b1))  # 785x40
    W1b1W2 = np.vstack((W1b1, W2.T))  # 795x40
    b2z = np.hstack((b2, np.zeros(NUM_HIDDEN - NUM_OUTPUT)))  # 40x1
    packed = np.vstack((W1b1W2, b2z))  # 796x40
    return packed


# Load the images and labels from a specified dataset (train or test).
def loadData(which):
    images = np.load("mnist_{}_images.npy".format(which))
    labels = np.load("mnist_{}_labels.npy".format(which))
    return images, labels


def plotSGDPath(trainX, trainY, ws):
    def toyFunction (x1, x2):
        return np.sin((2 * x1**2 - x2) / 10.)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    axis2 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    Yaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    Zaxis = toyFunction(Xaxis, Yaxis)
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE(X, Y, w):
    W1, b1, W2, b2 = unpack(w)

    # TODO
    # copied from hw3
    prod = np.multiply(X, np.log(Y))
    k_summed = prod.sum(axis=1)
    return -np.average(k_summed)
    # return cost


# convenience helper to calculate z1 or z2 during prediction
def calc_z(w, x, b):
    return w.dot(x) + b


def relu(x):
    return np.maximum(x, np.zeros(x.shape))


def relu_prime(x):
    return 1 if x > 0 else 0


def softmax(x):
    print(x.shape)
    exp = np.exp(x)
    sum = np.sum(exp, axis=1)
    print(sum.shape)
    return np.divide(exp, sum)


def calc_prediction(X, w):
    W1, b1, W2, b2 = unpack(w)
    # 784x40, 40x1, 40x10, 10x1
    z1 = calc_z(W1, X, b1)  # 40*N
    h1 = relu(z1)  # 40*N
    z2 = calc_z(W2, h1, b2)
    y_hat = softmax(z2)
    return z1, h1, y_hat


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE(X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    z1, h1, y_hat = calc_prediction(X, w)
    first_component = np.transpose(y_hat - Y).dot(W2)
    second_component = relu_prime(z1.T)
    g_t = np.multiply(first_component, second_component)

    grad_W2 = np.dot(y_hat - Y, h1.T)
    grad_b2 = y_hat - Y
    grad_W1 = np.dot(g_t.T, X.T)
    grad_b1 = g_t.T
    return pack(grad_W1, grad_b1, grad_W2, grad_b2)


# Given training and testing datasets and an initial set of weights/biases b,
# train the NN. Then return the sequence of w's obtained during SGD.
def train(epochs, batch_size, epsilon, trainX, trainY, testX, testY, w):
    train_fused = np.hstack((trainX, trainY))
    np.random.shuffle(train_fused)
    train_images = train_fused[:, :-10]
    train_values = train_fused[:, -10:]

    for epoch in range(epochs):
        for round in range(math.ceil(train_images.shape[0] / batch_size)):
            sample_img = train_images[round * batch_size:(round + 1) *
                                                         batch_size]
            sample_val = train_values[round * batch_size:(round + 1) *
                                                         batch_size]
            grad = gradCE(sample_img, sample_val, w)
            w = w - epsilon * grad

    print("fCE:", fCE(trainX, trainY, w))
    return w


def findBestHyperparameters():
    # hidden layer, learning rate, minibatch size, epochs, regularization strength
    tests = [
        [40, 0.01, 50, 2, 0.1],
        [40, 0.1, 25, 5, 0.01],
        [40, 0.001, 50, 10, 0.001],
        [40, 0.05, 75, 5, 0.001],
        [50, 0.01, 75, 5, 0.01],
        [50, 0.1, 25, 10, 0.1],
        [50, 0.001, 50, 5, 0.1],
        [30, 0.01, 25, 5, 0.01],
        [30, 0.1, 50, 5, 0.1],
        [30, 0.001, 75, 10, 0.001],
    ]

    validationX, validationY = loadData("validation")

    for test in tests:
        hidden_layer, learning_rate, minibatch_size, epoch_count, regularization_str = test
        # apply settings as necessary
        W1 = 2 * (np.random.random(size=(NUM_INPUT, hidden_layer)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
        b1 = 0.01 * np.ones(hidden_layer)
        W2 = 2 * (np.random.random(size=(hidden_layer, NUM_OUTPUT)) / hidden_layer ** 0.5) - 1. / hidden_layer ** 0.5
        b2 = 0.01 * np.ones(NUM_OUTPUT)

        w = pack(W1, b1, W2, b2)

        ws = train(epoch_count, minibatch_size, learning_rate, trainX, trainY, validationX, validationY, w)
        # ce = fCE(validationX, validationY, w)


if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_INPUT, NUM_HIDDEN))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)

    w = pack(W1, b1, W2, b2)

    W10, b10, W20, b20 = unpack(w)
    assert np.array_equal(W10, W1)
    assert np.array_equal(b10, b1)
    assert np.array_equal(W20, W2)
    assert np.array_equal(b20, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    w))

    # Train the network and obtain the sequence of w's obtained using SGD.
    ws = train(5, 32, 0.01, trainX, trainY, testX, testY, w)

    # Plot the SGD trajectory
    plotSGDPath(trainX, trainY, ws)
