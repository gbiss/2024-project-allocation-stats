from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from fair_stats import (
    Correlation,
    mBetaApprox,
    mBetaExact,
    Mean,
    Shape,
)


def bernoulli_samples(theta, n, m):
    return np.hstack(
        [stats.bernoulli(theta[i]).rvs(n).reshape((n, 1)) for i in range(m)]
    )


def infer(thetas, m, n=1):
    R = Correlation(m)
    nu = Shape(1)
    mu = Mean(m)
    mbeta = mBetaApprox(R, mu, nu)
    for i in range(thetas.shape[0]):
        bernoullis = bernoulli_samples(thetas[i, :], n, m)
        mbeta.update(bernoullis)

    return mbeta


m = 2
n = 100

# generate data from exact mBeta
plt.figure()
plt.xlim((0, 1))
plt.ylim((0, 1))
eps = 1
gamma = eps * np.ones((2**m,))
gamma[1] = 100
gamma[2] = 100
mbeta_e = mBetaExact(gamma)
theta_es = mbeta_e.sample(n)
plt.scatter(theta_es[:, 0], theta_es[:, 1], c="r", alpha=0.25)

mbeta_a = infer(theta_es, m)
theta_as = mbeta_a.sample(n)
plt.scatter(theta_as[:, 0], theta_as[:, 1], c="b", alpha=0.25)
plt.legend(["Exact mBeta", "Approx mBeta"], loc="best")

# generate extremal synthetic data
plt.figure()
plt.xlim((0, 1))
plt.ylim((0, 1))
thetas = []
for i in range(n):
    thetas.append(np.array([[0.99, 0.99], [0.01, 0.01]]))
theta_es = np.vstack(thetas)
plt.scatter(theta_es[:, 0], theta_es[:, 1], c="r", alpha=0.25)

mbeta_a = infer(theta_es, m)
theta_as = mbeta_a.sample(n)
plt.scatter(theta_as[:, 0], theta_as[:, 1], c="b", alpha=0.25)
plt.legend(["Synthetic samples", "Approx mBeta"], loc="best")

plt.show()
