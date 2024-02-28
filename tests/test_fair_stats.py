import numpy as np


def test_aggregates_not_enough_for_U():
    """Survey aggregates are not enough to form update matrix U

    This test provides a counterexample to the idea that, for the purpose of computing
    update matrix U, one can treat the surveyed values, 0-n for each of m courses, as n
    *simultaneous* draws from m Bernoulli distributions. The counterexample shows that
    two different sets of n flips will, having the same aggregates, will produce two
    different update matrices.
    """
    # example survey outcomes; each outcome has n=2 sets of m=3 coin flips
    outcomes_v1 = np.array([[1, 1, 1], [0, 1, 0]])
    outcomes_v2 = np.array([[1, 1, 0], [0, 1, 1]])

    n = outcomes_v1.shape[0]

    # aggregates of the flips are equal
    np.testing.assert_array_equal(
        np.sum(outcomes_v1, axis=0), np.sum(outcomes_v2, axis=0)
    )

    # transformation matrix (bit vector to categorical)
    H3 = np.array(
        [[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]]
    )

    # 2^m = 8
    M = H3.shape[1]

    # form d vectors: sum of basis vectors corresponding to column indices of H3 that match
    # rows of the outcomes vectors
    d_v1 = np.zeros((1, M))
    for row in range(n):
        h_index_v1 = np.where((H3 == outcomes_v1[row][:, None]).all(axis=0))[0][0]
        d_v1 += np.eye(1, M, h_index_v1)

    d_v2 = np.zeros((1, M))
    for row in range(n):
        h_index_v2 = np.where((H3 == outcomes_v2[row][:, None]).all(axis=0))[0][0]
        d_v2 += np.eye(1, M, h_index_v2)

    # form Delta matrices: diagonal matrix formed from d
    # pre- and post-multiply Delta by H3 to form U matrices
    U_v1 = H3 @ np.diag(d_v1.reshape((M,))) @ H3.T
    U_v2 = H3 @ np.diag(d_v2.reshape((M,))) @ H3.T

    # show that the two U matrices differ
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(U_v1, U_v2)
