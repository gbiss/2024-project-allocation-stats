from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
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


def plot(data1, data2, proj=False):
    if proj:
        pca = PCA(n_components=2)
        data = pca.fit_transform(np.vstack([data1, data2]))
        data1 = data[: data1.shape[0], :]
        data2 = data[data1.shape[0] :, :]

    plt.xlim((-1 * proj, 1))
    plt.ylim((-1 * proj, 1))
    plt.scatter(data1[:, 0], data1[:, 1], c="r", alpha=0.25)
    plt.scatter(data2[:, 0], data2[:, 1], c="b", alpha=0.25)


m = 2
n = 100
PROJ = True

# generate data from exact mBeta
plt.figure()
eps = 1
gamma = eps * np.ones((2**m,))
gamma[1] = 100
gamma[2] = 100
mbeta_e = mBetaExact(gamma)
theta_es = mbeta_e.sample(n)

mbeta_a = infer(theta_es, m)
theta_as = mbeta_a.sample(n)
plot(theta_es, theta_as, PROJ)
plt.legend(["Exact mBeta", "Approx mBeta"], loc="best")

# generate extremal synthetic data
plt.figure()
thetas = []
for i in range(n):
    thetas.append(np.array([[0.99] * m, [0.01] * m]))
theta_es = np.vstack(thetas)

mbeta_a = infer(theta_es, m)
theta_as = mbeta_a.sample(n)
plot(theta_es, theta_as, PROJ)
plt.legend(["Synthetic samples", "Approx mBeta"], loc="best")

plt.show()
