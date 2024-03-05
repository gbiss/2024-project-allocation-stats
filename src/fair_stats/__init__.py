import numpy as np

from fair_stats.survey import Corpus


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
