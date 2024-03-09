import numpy as np

from fair_stats.survey import Corpus


def binary(integer: int, n: int) -> np.ndarray:
    """Convert integer to binary

    Args:
        integer (int): Postive integer for conversion
        n (int): Number of bits

    Raises:
        OverflowError: Bits should be sufficient to encode integer

    Returns:
        np.ndarray: Bits comprising integer
    """
    if integer >= 2**n:
        raise OverflowError("index requires more than n bits")

    binary_str = format(integer, "b").zfill(n)

    return np.array(list(binary_str), dtype=int)


def integer(bits: np.ndarray) -> int:
    """Convert bit array to integer

    Args:
        bits (np.ndarray): Bit array

    Returns:
        int: Integer encoding of bit array
    """
    bits_str = "".join(str(bit) for bit in bits.flatten())
    integer = int(bits_str, 2)

    return integer


def transformation(n: int) -> np.ndarray:
    """Transformation matrix H (binary to categorical)

    Args:
        n (int): Number of bits

    Returns:
        np.ndarray: n X 2**n matrix with columns encoding ints 0 to 2**n-1
    """
    columns = []
    for i in range(2**n):
        columns.append(binary(i, n))

    return np.vstack(columns).T


def transform(bernoullis: np.ndarray) -> int:
    """Find index for bernoulis as column in H

    Args:
        bernoullis (np.ndarray): Column in H

    Returns:
        int: Index of column in H that matches bernoullis
    """
    return integer(bernoullis)


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

    def direct(self, H: np.ndarray) -> np.ndarray:
        """Directly compute U from H

        H is an m X 2**m transformation matrix. U is an m X m
        update matrix.

        Args:
            H (np.ndarray): Transformation matrix (bit vector to categorical)

        Returns:
            np.ndarray: Update matrix U
        """
        n, _ = self.bernoullis.shape
        w = H.shape[1]
        Delta = np.zeros((w, w))
        for row in range(n):
            h_index = transform(self.bernoullis[row][:, None])
            Delta[h_index, h_index] += 1

        return H @ Delta @ H.T

    def indirect(self) -> np.ndarray:
        m = self.bernoullis.shape[1]
        U = np.zeros((m, m))
        for row in range(self.bernoullis.shape[0]):
            bits = self.bernoullis[row, :]
            U += np.outer(bits, bits)

        return U


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
