import numpy as np


def find_cost(predict, actual):
    diff_squared = np.square(predict - actual)
    return np.average(diff_squared) / 2


def find_cost_with_penalty(predict, actual, weight, alpha):
    diff_squared = np.square(predict - actual)
    old_cost = np.average(diff_squared) / 2
    penalty_term = (alpha / (2*len(actual)))*weight*np.transpose(weight)
    return old_cost + penalty_term

    
def calc_prediction(X, aug_w):
    w = aug_w[:-1]
    b = aug_w[-1]
    pred = X.dot(w) + b
    return pred

    
    
def gradient(X, y, w, b):
    diff = X.dot(w) + b - y
    return np.transpose(X).dot(np.transpose(diff)) / X.shape[0], np.average(diff)

def predicts(train_images, test_images, train_values, test_values, aug_w):
    train_pred = calc_prediction(train_images, aug_w)
    test_pred = calc_prediction(test_images, aug_w)
    train_cost = find_cost(train_pred, train_values)
    test_cost = find_cost(test_pred, test_values)
    print("Analytical: \n")
    print("Train cost: ", train_cost)
    print("Test cost: ", test_cost)
    

def augment(X):
    return np.hstack((X, np.transpose(np.atleast_2d(np.ones(X.shape[0])))))


def one_shot(train_images, test_images, train_values, test_values):
    aug_train_images = augment(train_images)
    A = np.transpose(aug_train_images).dot(aug_train_images)
    b = np.transpose(aug_train_images).dot(np.transpose(train_values))
    aug_w = np.linalg.solve(A,b)
    predicts(train_images, test_images, train_values, test_values, aug_w)
    
def gradiant_descent(T, e, train_images, test_images, train_values, test_values):
    aug_w = generate_weights()
    w = aug_w[:-1]
    b = aug_w[-1]
    for t in range(T):
        
        g, delta = gradient(train_images, train_values, w, b)
        w = w - e*g
        b = b - e*delta
        print(np.average(g))
        
    aug_w = np.hstack((w,b))
    predicts(train_images, test_images, train_values, test_values, aug_w)


def generate_weights():
    sigma = 0.01 ** 0.5
    mu = 0.5
    return sigma * np.random.randn(48*48+1) + mu


if __name__ == "__main__":
    train_images = np.load("age_Xtr.npy").reshape(-1, 2304)
    test_images = np.load("age_Xte.npy").reshape(-1, 2304)
    train_values = np.load("age_ytr.npy")
    test_values  = np.load("age_yte.npy")
    one_shot(train_images, test_images, train_values, test_values)
    gradiant_descent(5000, 0.003, train_images, test_images, train_values, test_values)