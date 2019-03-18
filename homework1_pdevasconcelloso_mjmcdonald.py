import numpy as np


def problem1(A, B):
    return A + B


def problem2(A, B, C):
    return np.dot(A, B) - C


def problem3(A, B, C):
    return np.multiply(A, B) + np.transpose(C)


def problem4(x, y):
    return np.inner(x, y)


def problem5(A):
    return np.zeros(A.shape)


def problem6(A):
    return np.ones(A.shape[0])


def problem7(A, alpha):
    return A + alpha * np.eye(A.shape)


def problem8(A, i, j):
    return A[i - 1][j - 1]


def problem9(A, i):
    return np.sum(A[i])


def problem10(A, c, d):
    return ...


def problem11(A, k):
    return ...


def problem12(A, x):
    return ...


def problem13(A, x):
    return ...
