from fair_stats import aggregate
import numpy as np


def test_aggregates_not_enough_for_U(H3):
    """Survey aggregates are not enough to form update matrix U

    This test provides a counterexample to the idea that, for the purpose of computing
    update matrix U, one can treat the surveyed values, 0-n for each of m courses, as n
    *simultaneous* draws from m Bernoulli distributions. The counterexample shows that
    two different sets of n flips will, having the same aggregates, will produce two
    different update matrices.
    """
    # example survey outcomes; each outcome has n=2 sets of m=3 coin flips
    bernoullis1 = np.array([[1, 1, 1], [0, 1, 0]])
    bernoullis2 = np.array([[1, 1, 0], [0, 1, 1]])
    w = 2 ** bernoullis1.shape[1]

    # aggregates of the flips are equal
    np.testing.assert_array_equal(
        np.sum(bernoullis1, axis=0), np.sum(bernoullis2, axis=0)
    )

    # form d vectors: sum of basis vectors corresponding to column indices of H3 that match
    # rows of the outcomes vectors
    d_v1 = aggregate(bernoullis1, H3)
    d_v2 = aggregate(bernoullis2, H3)

    # form Delta matrices: diagonal matrix formed from d
    # pre- and post-multiply Delta by H3 to form U matrices
    U_v1 = H3 @ np.diag(d_v1.reshape((w,))) @ H3.T
    U_v2 = H3 @ np.diag(d_v2.reshape((w,))) @ H3.T

    # show that the two U matrices differ
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(U_v1, U_v2)