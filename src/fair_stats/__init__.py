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
    """Data matrix U"""

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
        """Calculate U without materializing H

        Returns:
            np.ndarray: Update matrix
        """
        m = self.bernoullis.shape[1]
        U = np.zeros((m, m))
        for row in range(self.bernoullis.shape[0]):
            bits = self.bernoullis[row, :]
            U += np.outer(bits, bits)

        return U


class Shape:
    """Shape parameter nu"""

    def __init__(self, nu: float) -> None:
        """Prior for nu

        Args:
            nu (float): Prior shape parameter
        """
        self._data = nu

    def update(self, n: int) -> "Shape":
        """Posterior for nu

        Args:
            n (int): Number of Bernoulli sample vectors

        Returns:
            Shape: Updated shape object
        """
        self._data += n

    def __call__(self) -> float:
        """Value of shape

        Returns:
            float: Shape parameter
        """
        return self._data


def standard_deviations(mu: "Mean", nu: Shape) -> np.ndarray:
    """Diagonal V matrix of standard deviations

    Args:
        mu (Mean): mX1 mean vector
        nu (Shape): scalar shape parameter

    Returns:
        np.ndarray: mXm matrix with standard deviations along diagonal
    """
    return np.diag(np.multiply(mu(), 1 - mu()).flatten()) / (nu() * (nu() + 1))


class Moment:
    """Moment matrix A"""

    def __init__(self, Sigma: np.ndarray, mu: "Mean", nu: Shape) -> None:
        """Prior moment matrix A

        Args:
            Sigma (np.ndarray): Covariance matrix
            mu (Mean): Mean vector
            nu (Shape): Shape parameter
        """
        self._data = nu() * ((nu() + 1) * Sigma() @ np.outer(mu(), mu()))

    def update(self, U: Update) -> "Moment":
        """Posterior moment matrix

        Args:
            U (Update): Update (data) matrix

        Returns:
            Moment: Updated Moment object
        """
        self._data += U.indirect()

    def __call__(self) -> np.ndarray:
        """Value of moment matrix

        Returns:
            np.ndarray: Moment matrix
        """
        return self._data


class Mean:
    """Mean vector mu"""

    def __init__(self, m: int) -> None:
        """Prior for mean vector mu

        Args:
            m (int): Number of dimensions of each Bernoulli sample
        """
        self.m = m
        self._data = np.ones((self.m, 1)) / 2

    def update(self, A: Moment, nu: Shape) -> "Mean":
        """Posterior for mean vector mu

        Args:
            A (np.ndarray): Moment object
            nu (float): Shape object

        Returns:
            Mean: Updated mean vector
        """
        self._data = np.diag(A()) / nu()

        return self

    def __call__(self) -> np.ndarray:
        """Value of mean vector

        Returns:
            np.ndarray: Mean vector
        """
        return self._data


class Covariance:

    def __init__(self, R: np.ndarray, V: np.ndarray) -> None:
        """Prior covariance matrix Sigma

        Args:
            R (np.ndarray): mXm correlation matrix
            V (np.ndarray): mXm standard deviations matrix
        """
        self._data = V**0.5 @ R @ V**0.5

    def update(self, A: Moment, mu: Mean, nu: Shape) -> "Covariance":
        """Posterior covariance Matrix Sigma

        Args:
            A (Moment): Moment matrix
            mu (Mean): Mean vector
            nu (Shape): Shape parameter

        Returns:
            Covariance: Updated covariance matrix
        """
        return (A() / nu() - np.outer(mu(), mu())) / (nu() + 1)

    def __call__(self):
        """Value of covariance matrix

        Returns:
            np.ndarray: Covariance matrix
        """
        return self._data


def correlation(V: np.ndarray, Sigma: Covariance) -> np.ndarray:
    """R matrix

    Args:
        V (np.ndarray): mXm standard deviation matrix
        Sigma (Covariance): Covariance object containing mXm matrix

    Returns:
        np.ndarray: mXm correlation matrix
    """
    V_inv = np.linalg.inv(V)

    return V_inv**0.5 @ Sigma() @ V_inv**0.5
