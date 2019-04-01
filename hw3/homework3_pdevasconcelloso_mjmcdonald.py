import numpy as np
import matplotlib.pyplot as plt


def find_cost(predict, actual, n):
    prod = np.multiply(actual, np.transpose(np.log(predict)))
    k_summed = prod.sum(axis=1)
    return np.average(k_summed)
    
def calc_prediction(aug_X, aug_w):
    z = np.exp(aug_X.dot(np.transpose(aug_w)))
    total = z.sum(axis=1)
    pred = np.transpose(z) / total.flatten()
    return np.transpose(pred)

    
def gradient(X, y, w, b):
    diff = X.dot(w) + b - y
    return np.transpose(X).dot(np.transpose(diff)) / X.shape[0], np.average(diff)




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



def gradient_descent(T, e, train_images, test_images, train_values, test_values):
    aug_w = generate_weights()
    w = aug_w[:-1]
    b = aug_w[-1]
    for t in range(T):
        
        g, delta = gradient(train_images, train_values, w, b)
        w = w - e*g
        b = b - e*delta
        
    aug_w = np.hstack((w,b))
    return predicts(train_images, test_images, train_values, test_values, aug_w, "gradient_descent")


def generate_weights():
    sigma = 0.01 ** 0.5
    mu = 0.5
    return sigma * np.random.randn(48*48+1) + mu


if __name__ == "__main__":
    train_images = np.load("age_Xtr.npy").reshape(-1, 2304)
    test_images = np.load("age_Xte.npy").reshape(-1, 2304)
    train_values = np.load("age_ytr.npy")
    test_values  = np.load("age_yte.npy")
    # m1train, m1test = one_shot(train_images, test_images, train_values, test_values)
    # m2train, m2test = gradient_descent(5000, 0.003, train_images, test_images, train_values, test_values)
    m3train, m3test = gradient_descent_regularized(5000, 0.003, 0.1, train_images, test_images, train_values, test_values)
    show_five_worst(m3test, test_values, test_images)
