import math
import numpy as np
import scipy.spatial.distance as dist


def compute_EucDis(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def compute_ManDis(x, y):
    return np.sum(np.abs(x - y))


def compute_CheDis(x, y):
    return np.max(np.abs(x - y))


def compute_MinDis(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def compute_CosineDis(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * (np.linalg.norm(y)))


def compute_HamMingDis(x, y):
    return np.shape(np.nonzero(x - y)[0])[0]


def compute_JacDis(x, y):
    return dist.pdist(np.array([x, y]), 'jaccard')


if __name__ == '__main__':
    x = [1, 1, 1, 1, 1]
    y = [1, 1, 4, 1, 0]
    print compute_JacDis(np.array(x), np.array(y))
