import numpy as np
import torch
from matplotlib import pyplot as plt


def batch_euclidean_dist(x, y):
    """

    :param x:
    :param y:
    :return:
    """

    B1, m, d = x.shape
    B2, n, d = y.shape
    xx = np.sum(x ** 2, axis=-1, keepdims=True).repeat(n, axis=2)
    xx = np.expand_dims(xx, 1).repeat(B2, axis=1)
    yy = np.sum(y ** 2, axis=-1, keepdims=True).repeat(m, axis=2)
    yy = np.expand_dims(yy, 0).repeat(B1, axis=0)
    dist = xx + yy
    dist -= 2 * np.matmul(np.expand_dims(x, 1).repeat(B2, axis=1),
                          np.expand_dims(np.transpose(y, (0, 2, 1)), 0).repeat(B1, axis=0))
    return dist


def shortest_dist(dist_mat):
    m, n = dist_mat.size()[:2]
    dist = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i][j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
            else:
                dist[i][j] = torch.min(torch.min(dist[i - 1][j], dist[i][j - 1]),
                                       dist[i - 1][j - 1]) + dist_mat[i, j]
    plt.imshow(np.array(dist).astype(np.float))
    plt.title("shortest distance matrix")
    plt.show()
    return dist[-1][-1]


def batch_local_dist(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    assert len(x.shape) == 3
    assert len(y.shape) == 3
    assert x.shape[0] == y.shape[0]
    assert x.shape[-1] == y.shape[-1]

    # shape [N, m, n]
    dist_mat = batch_euclidean_dist(x, y)
    plt.imshow(np.array(dist_mat[0, ...]).astype(np.float))
    plt.title("distance matrix")
    plt.show()
    # dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
    print(dist_mat)
    # shape [N]
    dist = shortest_dist(dist_mat.permute(1, 2, 0))
    return dist


if __name__ == '__main__':
    x = np.random.randn(3, 16, 32)
    y = np.random.randn(100, 16, 32)
    dist_mat = batch_euclidean_dist(x, y)
    # dist_ap, dist_an, p_inds, n_inds = hard_example_mining(dist_mat,return_inds=True)
    # from IPython import embed
    # embed()
