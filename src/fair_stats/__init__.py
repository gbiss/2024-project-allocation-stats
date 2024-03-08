import numpy as np

from fair_stats.survey import Corpus


def transform(bernoullis: np.ndarray, H: np.ndarray) -> int:
    """Find index for bernoulis as column in H

    Args:
        bernoullis (np.ndarray): Column in H
        H (np.ndarray): Transformation matrix (bit vector to categortical)

    Returns:
        int: Index of column in H that matches bernoullis
    """
    return np.where((H == bernoullis).all(axis=0))[0][0]


def aggregate(bernoullis: list[np.ndarray], H: np.ndarray) -> np.ndarray:
    """d vector

    A vector of categorical aggregates for m variables over n trials.

    Args:
        bernoullis (list[np.ndarray]): n X m matrix of Bernoulli variates
        H (np.ndarray): m-dimensional transformation matrix (bit vector to categorical)

    Returns:
        np.ndarray: 2**m entries, each a count of configurations of m variable outomes
    """
    n, m = bernoullis.shape
    w = 2**m
    d = np.zeros((1, w))
    for row in range(n):
        h_index = transform(bernoullis[row][:, None], H)
        d += np.eye(1, w, h_index)

    return d


class Update:
    """U matrix"""

    def __init__(self, bernoullis: list[np.ndarray]):
        self.bernoullis = bernoullis

    def direct(self, H):
        w = H.shape[1]
        Delta = np.zeros((w, w))
        return H @ Delta @ H.T


class Covariance:
    """Sigma matrix"""

    pass


class Correlation:
    """R matrix"""

    def __init__(self, m: int) -> None:
        self.m = m
        self._data = np.eye(self.m)

    def update(self, V: np.ndarray, Sigma: np.ndarray) -> "Correlation":
        V_inv = np.linalg.inv(V)
        self._data = V_inv**0.5 @ Sigma @ V_inv**0.5

        return self


class StandardDeviations:
    """Diagonal matrix V of standard deviations"""

    pass


class Mean:
    """Mean vector mu"""

    def __init__(self, m: int) -> None:
        self.m = m
        self._data = np.ones((self.m, 1)) / 2

    def update(self, A: np.ndarray, nu: float) -> "Mean":
        self._data = np.diag(A) / nu

        return self


class Data:
    """Data matrix U"""

    def __init__(self, m: int, corpus: Corpus) -> None:
        self.m = m
        self.corpus = corpus
