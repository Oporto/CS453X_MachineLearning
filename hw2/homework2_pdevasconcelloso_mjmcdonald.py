import numpy as np
import matplotlib.pyplot as plt


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
    diff = X.dot(w) + b - y
    return np.transpose(X).dot(np.transpose(diff)) / X.shape[0], np.average(diff)


def gradient_regularized(X, y, w, b, alpha):
    diff = X.dot(w) + b - y
    unregularized = np.transpose(X).dot(np.transpose(diff)) / X.shape[0]
    return unregularized + (alpha / X.shape[0])*w, np.average(diff)


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


def predicts_regularized(train_images, test_images, train_values, test_values, aug_w, alpha):
    train_pred = calc_prediction(train_images, aug_w)
    test_pred = calc_prediction(test_images, aug_w)
    train_cost = find_cost_with_penalty(train_pred, train_values, aug_w, alpha)
    test_cost = find_cost_with_penalty(test_pred, test_values, aug_w, alpha)
    np.save("regularized", aug_w)
    print("Regularized Gradient Descent: \n")
    print("Train cost: ", train_cost)
    print("Test cost: ", test_cost)
    
    print("Mean square error: ", find_rmse(test_pred, test_values))

    im = aug_w[0:-1]
    im = im.reshape(48, 48)
    plt.imshow(im, cmap='gray')
    plt.title("regularized")
    plt.show()
    return train_pred, test_pred
    

def augment(X):
    return np.hstack((X, np.transpose(np.atleast_2d(np.ones(X.shape[0])))))


def one_shot(train_images, test_images, train_values, test_values):
    aug_train_images = augment(train_images)
    A = np.transpose(aug_train_images).dot(aug_train_images)
    b = np.transpose(aug_train_images).dot(np.transpose(train_values))
    aug_w = np.linalg.solve(A,b)
    return predicts(train_images, test_images, train_values, test_values, aug_w, "one_shot")


def gradient_descent(T, e, train_images, test_images, train_values, test_values):
    aug_w = generate_weights()
    w = aug_w[:-1]
    b = aug_w[-1]
    for t in range(T):
        
        g, delta = gradient(train_images, train_values, w, b)
        w = w - e*g
        b = b - e*delta
        print(t,"/",T)
        
    aug_w = np.hstack((w,b))
    return predicts(train_images, test_images, train_values, test_values, aug_w, "gradient_descent")


def gradient_descent_regularized(T, e, alpha, train_images, test_images, train_values, test_values):
    aug_w = generate_weights()
    #record_w = np.empty([500,2304])
    #deltas = np.empty([500,1])
    w = aug_w[:-1]
    b = aug_w[-1]
    for t in range(T):
        g, delta = gradient_regularized(train_images, train_values, w, b, alpha)
        '''if (t + 1) % 10 == 0 or t == 0:
            record_w[(t+1)//10] = w
            deltas[(t+1)//10] = delta
            print(np.average(g), delta)'''
        w = w - e * g
        b = b - e * delta
        print("reg", t,"/",T)

        
    #np.save("weights_reg", record_w)
    #np.save("deltas_reg", deltas)
    
    aug_w = np.hstack((w, b))
    return predicts_regularized(train_images, test_images, train_values, test_values, aug_w, alpha)


def generate_weights():
    sigma = 0.01 ** 0.5
    mu = 0.5
    return sigma * np.random.randn(48*48+1) + mu


def show_five_worst(pred_values, test_values, test_images):
    for i in range(5):
        error = np.abs(test_values - pred_values)
        index = np.argmax(error)
        im = test_images[index]
        im = im.reshape(48, 48)
        plt.imshow(im, cmap='gray')
        plt.title("worst #"+str(i+1)+": predicted "+str(pred_values[index])+" but was "+str(test_values[index]))
        plt.show()
        pred_values[index] = test_values[index] # so it is ignored next time


if __name__ == "__main__":
    train_images = np.load("age_Xtr.npy").reshape(-1, 2304)
    test_images = np.load("age_Xte.npy").reshape(-1, 2304)
    train_values = np.load("age_ytr.npy")
    test_values  = np.load("age_yte.npy")
    # m1train, m1test = one_shot(train_images, test_images, train_values, test_values)
    # m2train, m2test = gradient_descent(5000, 0.003, train_images, test_images, train_values, test_values)
    m3train, m3test = gradient_descent_regularized(5000, 0.003, 0.1, train_images, test_images, train_values, test_values)
    show_five_worst(m3test, test_values, test_images)
