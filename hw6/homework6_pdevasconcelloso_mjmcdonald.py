import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import IncrementalPCA
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
    # print(W1.shape, b1.shape, W2.shape, b2.shape)
    # 784x40, 40x1, 40x10, 10x1
    W1b1 = np.vstack((W1, b1))  # 785x40
    W1b1W2 = np.vstack((W1b1, W2.T))  # 795x40
    b2z = np.hstack((b2, np.zeros(W1.shape[1] - 10)))  # 40x1
    packed = np.vstack((W1b1W2, b2z))  # 796x40
    return packed


# Load the images and labels from a specified dataset (train or test).
def loadData(which):
    images = np.load("mnist_{}_images.npy".format(which)).T
    labels = np.load("mnist_{}_labels.npy".format(which))
    return images, labels

def findCost (trainX, trainY, ipca, x1, x2):
    w = ipca.inverse_transform(np.array([x1,x2])).reshape(796,40)
    return fCE(trainX,trainY,w)

def plotSGDPath(trainX, trainY, ws):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ipca = gen_PCA(ws)
    
    reduced_w = ipca.transform(ws.reshape(-1,796*40))
    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    axis2 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    axis1 = np.add(axis1, reduced_w[-1][0])
    axis2 = np.add(axis2, reduced_w[-1][1])
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    find_vect_cost = np.vectorize(findCost, excluded=[0,1,2])
    Zaxis = find_vect_cost(trainX, trainY, ipca, Xaxis, Yaxis)
    print(Xaxis.shape, Yaxis.shape)
    print(Xaxis[1,1])
    print(Zaxis.shape)
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis = reduced_w[:,0]
    Yaxis = reduced_w[:,1]
    Zaxis = np.apply_along_axis(lambda w: findCost(trainX, trainY, ipca, w[0], w[1]), 0, ws)
    print("Zaxis scatter:" + Zaxis.shape)
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE(X, Y, w):
    __, __, Y_hat = calc_prediction(X, w)
    prod = np.multiply(Y,np.log(Y_hat))
    k_summed = prod.sum(axis=1)
    return -1*np.average(k_summed)
    # return cost


# convenience helper to calculate z1 or z2 during prediction
def calc_z(w, x, b):
    return np.add(w.T.dot(x).T, b)


def relu(x):
    return np.maximum(x, np.zeros(x.shape))


def relu_prime(x):
    relu_prime_x = np.ones(x.shape)
    comp = x > 0
    return np.multiply(relu_prime_x, comp)


def softmax(x):
    exp = np.exp(x)
    sum = np.sum(exp, axis=1)
    return np.divide(exp.T, sum).T


def calc_prediction(X, w):
    W1, b1, W2, b2 = unpack(w)
    # 784x40, 40x1, 40x10, 10x1
    z1 = calc_z(W1, X, b1)  # 40*N
    h1 = relu(z1)  # 40*N
    z2 = calc_z(W2, h1.T, b2)
    y_hat = softmax(z2)
    return z1, h1, y_hat


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE(X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    z1, h1, y_hat = calc_prediction(X, w)
    first_component = np.dot(y_hat - Y, W2.T).T
    second_component = relu_prime(z1.T)
    g_t = np.multiply(first_component, second_component)

    grad_W2 = np.dot(np.transpose(y_hat - Y), h1)
    grad_b2 = np.average(y_hat - Y, axis=0)
    grad_W1 = np.dot(g_t, X.T)
    grad_b1 = np.average(g_t.T, axis=0)
    return pack(grad_W1.T, grad_b1.T, grad_W2.T, grad_b2.T)


# Given training and testing datasets and an initial set of weights/biases b,
# train the NN. Then return the sequence of w's obtained during SGD.
def train(epochs, batch_size, epsilon, trainX, trainY, testX, testY, w):
    train_fused = np.hstack((trainX.T, trainY))
    np.random.shuffle(train_fused)
    train_images = train_fused[:, :-10].T
    train_values = train_fused[:, -10:]
    ws = []

    for epoch in range(epochs):
        for round in range(math.ceil(train_images.shape[0] / batch_size)):
            sample_img = train_images[:,round * batch_size:(round + 1) *
                                                         batch_size]
            sample_val = train_values[round * batch_size:(round + 1) *
                                                         batch_size]
            grad = gradCE(sample_img, sample_val, w)
            
            w = w - epsilon * grad
        ws.append(w)

    print("fCE:", fCE(trainX, trainY, w))
    return np.array(ws)


def findBestHyperparameters():
    # hidden layer, learning rate, minibatch size, epochs, regularization strength
    test_results = []
    tests = []

    # for hidden in [30, 40, 50]:
    #     for learning in [0.1, 0.01, 0.05, 0.001, 0.005]:
    #         for batch in [25, 50, 75]:
    #             for epochs in [5, 10, 25, 50, 100]:
    #                 tests.append([hidden, learning, batch, epochs, 0.1])

    tests = []
    param1opts = [30,40,50]
    param2opts = [0.001,0.005,0.01,0.05,0.1]
    param3opts = [25,50,75]
    param4opts = [2,5,10,20]
    param5opts = [0.001,0.01,0.1]
    for p1 in param1opts:
        for p2 in param2opts:
            for p3 in param3opts:
                for p4 in param4opts:
                    for p5 in param5opts:
                        tests.append([p1,p2,p3,p4,p5])

    validationX, validationY = loadData("validation")

    for index, test in enumerate(tests):
        hidden_layer, learning_rate, minibatch_size, epoch_count, regularization_str = test
        # apply settings as necessary
        W1 = 2 * (np.random.random(size=(NUM_INPUT, hidden_layer)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
        b1 = 0.01 * np.ones(hidden_layer)
        W2 = 2 * (np.random.random(size=(hidden_layer, NUM_OUTPUT)) / hidden_layer ** 0.5) - 1. / hidden_layer ** 0.5
        b2 = 0.01 * np.ones(NUM_OUTPUT)

        w = pack(W1, b1, W2, b2)

        ws = train(epoch_count, minibatch_size, learning_rate, trainX, trainY, validationX, validationY, w)
        ce = fCE(validationX, validationY, w)
        print("test:",test)
        print("ce:",ce)

        test_results.append((test, ce))

    best_config = []
    best_score = 100000
    for test in test_results:
        config, ce = test
        if ce < best_score:
            best_score = ce
            best_config = config
    print("Best config:", best_config)
    print("fCE:", best_score)
    return best_config, best_score

def gen_PCA(w_array):
    ipca = IncrementalPCA(n_components=2)
    ipca.fit(w_array.reshape(-1,796*40))
    return ipca

def w_vary(w,x1,x2):
    w[0]+=x1
    w[1]+=x2
    return w



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
    print(w.shape)

    W10, b10, W20, b20 = unpack(w)
    assert np.array_equal(W10, W1)
    assert np.array_equal(b10, b1)
    assert np.array_equal(W20, W2)
    assert np.array_equal(b20, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    
    # print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[idxs,:]), w_),
    #                                 lambda w_: gradCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[idxs,:]), w_),
    #                                 w))
    
    # Train the network and obtain the sequence of w's obtained using SGD.
    ws = train(50, 50, 0.001, trainX, trainY, testX, testY, w)

    # best result: [hidden=40, epsilon=0.1, batch_size=25, epochs=5, regularization=0.01]
    best_config, best_score = findBestHyperparameters()

    # Plot the SGD trajectory
    plotSGDPath(trainX, trainY, ws)

    print("test fCE:", fCE(testX, testY, ws))

