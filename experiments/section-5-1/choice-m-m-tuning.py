import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
import pandas as pd
np.random.seed(1)

# Load package
import os
import sys
try: sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError: print("Cannot load testing module")
from resample_and_test import ShiftTester
np.random.seed(1)

def e(n=1, mu=0): 
    return np.random.normal(size=(n, 1), loc=mu)  # Gaussian noise
def cb(*args): 
    return np.concatenate(args, axis=1)  # Col bind
p = norm.pdf

# Model parameters
sigma_X = 1.0
sigma_eps = 2.0
n = 10000

# Simulate data from a Gaussian SCM
def scm(n, causal_effect=0):
    X1 = sigma_X*e(n)
    X2 = X1 + sigma_eps*e(n)
    X3 = causal_effect*X1 + X2 + e(n)
    return cb(X1, X2, X3)

# Weight function
def weight(X, scale=1): 
    return p(X[:, 1], scale=scale, loc=0)/p(X[:, 1], loc=X[:,0], scale=sigma_eps)

# Test function
def T(X): 
    return 1*(sm.OLS(X[:, 2], sm.tools.add_constant(X[:, 0])).fit().pvalues[1] < 0.05)

# Compute values m that would be used by conservative m-tuning
m_vals = []
for c_e in [0, 0.4]:
    for _ in range(100):
        psi = ShiftTester(weight, T, replacement=False, verbose=False)
        X = scm(n, causal_effect=c_e)
        w = weight(X)
        K = np.mean(w**2)
        m_ = psi.maximize_m_bound(K, n, alpha_P=0.05, alpha=0.1)
        m_vals.append({'m': m_, 'causal_effect': c_e})
df = pd.DataFrame(m_vals)
df.to_csv("experiments/section-5-1/choice-m-m-tuning.csv")