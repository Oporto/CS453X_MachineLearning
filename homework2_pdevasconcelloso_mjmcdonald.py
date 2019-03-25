import numpy as np


def find_cost(predict, actual):
    diff_squared = np.square(predict - actual)
    return np.average(diff_squared) / 2
    
def predict_one(x, w, b):
    return np.transpose(x).dot(w) + b
    
def calc_prediction(X, aug_w):
    w = aug_w[:-1]
    b = aug_w[-1]
    pred = np.apply_along_axis(lambda x: np.transpose(x).dot(w) + b, 0, X)
    return pred
    
def augment(X):
    return np.vstack(X, np.ones(X.shape[0]))

def one_shot(train_images, test_images, train_values, test_values):
    aug_train_images = augment(train_images)
    A = aug_train_images.dot(np.transpose(aug_train_images))
    b = aug_train_images.dot(train_values)
    aug_w = np.linalg.solve(A,b)
    train_pred = calc_prediction(train_images, aug_w)
    test_pred = calc_prediction(test_images, aug_w)
    train_cost = find_cost(train_pred, train_values)
    test_cost = find_cost(test_pred, test_values)
    print("Analytical: \n")
    print("Train cost: ", train_cost)
    print("Test cost: ", test_cost)


if __name__ == "__main__":
    train_images = np.load("age_Xtr").reshape(-1, 2304)
    test_images = np.load("age_Xte").reshape(-1, 2304)
    train_values = np.load("age_ytr")
    test_values  = np.load("age_yte")
    one_shot(train_images, test_images, train_values, test_values)