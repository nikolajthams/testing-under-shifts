import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import pandas as pd
from ananke.graphs import ADMG
from ananke.models import LinearGaussianSEM


np.random.seed(2)
n = 1000000

# Help functions
def e(n=1): return np.random.normal(size=(n, 1))  # Gaussian noise
def cb(*args): return np.concatenate(args, axis=1)  # Col bind
def rb(*args): return np.concatenate(args, axis=0)  # Col bind
V_wo = lambda i: list(np.delete(np.arange(4), i))
inv = np.linalg.inv

# Simulate data from a Gaussian SCM
def scm(n, causal_effect=0):
    H = e(n)
    X1 = e(n)
    X2 = X1 + H + e(n)
    X3 = X2 + 2*e(n)
    X4 = causal_effect*X1 + X3 + H + e(n)
    return cb(H, X1, X2, X3, X4)

X = scm(n, causal_effect=0)[:,1:].T

# Initialize
pa = {0: [], 1: [0], 2: [1], 3: [2]}
sp = {0: [], 1: [3], 2: [], 3: [1]}

B = np.zeros((4,4))
B[[1, 2, 3], [0, 1, 2]] = 1.0
Omega = np.diag([1.0, 2.0, 4.0, 2.0])
Omega[[1, 3], [3, 1]] = 1.0
# B = np.zeros((4,4))
# B[[1, 2, 3], [0, 1, 2]] = -1.0
# Omega = np.diag([1.0, 2.0, 3.0, 2.0])
# Omega[[1, 3], [3, 1]] = 0.3


for _ in tqdm(range(50)):
    B_old, Omega_old = B.copy(), Omega.copy()
    for i in range(4):
        eps = (np.eye(4) - B)@X
        Z = inv(Omega)[sp[i]][:,V_wo(i)]@eps[V_wo(i)]
        # Z = inv(Omega[V_wo(i)][:,V_wo(i)])@eps[V_wo(i)]
        # Z = Z[[j - 1*(i<j) for j in sp[i]]]

        regressor = rb(X[pa[i]], Z).T
        target = X[i].T
        if regressor.shape[1] > 0:
            results = sm.OLS(target, regressor).fit()
            cond_var = results.mse_resid
        else:
            cond_var = np.var(target)

        if regressor.shape[1] > 0:
            # if len(pa[i]) > 0:
            for idx, j in enumerate(pa[i]):
                B[i, j] = results.params[idx]
            # if len(sp[i]) > 0:
            for idx, j in enumerate(sp[i]):
                Omega[[j, i], [i, j]] = results.params[len(pa[i]) + idx]
        Omega[i, i] = cond_var + Omega[i, V_wo(i)]@inv(Omega)[V_wo(i)][:,V_wo(i)]@Omega[V_wo(i), i]

    diff = np.sum((B_old - B)**2 + (Omega_old - Omega)**2)
    if diff < 0.000001: break

Omega
(inv(np.eye(4)-B)@Omega@inv(np.eye(4)-B).T).round(1)
(1/n*X@X.T).round(1)


### USING ananke
vertices = ["A", "B", "C", "D"]
di_edges = [("A", "B"), ("B", "C"), ("C", "D")]
bi_edges = [("B", "D")]
G = ADMG(vertices, di_edges=di_edges, bi_edges=bi_edges)
model = LinearGaussianSEM(G, method="trust-exact")
data = pd.DataFrame({"A": X[0], "B": X[1], "C": X[2], "D": X[3]})
model.fit(data)
model.omega_
model.B_
