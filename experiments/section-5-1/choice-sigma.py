import numpy as np
from multiprocessing import Pool, cpu_count
import statsmodels.api as sm
from scipy.stats import norm
from tqdm import tqdm
from itertools import product
import pandas as pd

np.random.seed(1)

# Load testing module
import os
import sys
try: sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError: print("Cannot load testing module")
from resample_and_test import ShiftTester

# Help functions
def e(n=1, mu=0): return np.random.normal(size=(n, 1), loc=mu)  # Gaussian noise
def u(n=1): return 2*np.random.uniform(size=(n, 1))-1  # Gaussian noise
def cb(*args): return np.concatenate(args, axis=1)  # Col bind

# Gaussian density
p = norm.pdf

sigma_X = 1
sigma_eps = 2
def scm(n, causal_effect=0):
    X1 = sigma_X * e(n)
    X2 = X1 + sigma_eps*e(n)
    X3 = causal_effect*X1 + X2 + e(n)
    return cb(X1, X2, X3)

# Weight function
def weight(X, scale=1): return p(X[:, 1], scale=scale, loc=0)/p(X[:, 1], loc=X[:,0], scale=sigma_eps)

# Test function
def T(X): return sm.OLS(X[:, 2], sm.tools.add_constant(X[:, 0])).fit().pvalues[1]

# Target SDs (scales) to be used
theo_thres = np.sqrt(2*(sigma_eps**2 - sigma_X**2))
scales = [theo_thres*10**(x/5) for x in range(-5, 10)]
# Causal effect X1 -> X4. When 0, corresponds to no edge.
causal_effects = [0, 0.4]

# Replacement ranges
replace_range = []

n = 10000
## Wrap as function to apply multiprocessing
# 1 experiment simulates a data set and returns p-vals for different target SDs (scales)
def conduct_experiment(i=None):
    out = []
    for c_e in causal_effects:
        X = scm(n, causal_effect=c_e)
        for scale in scales:
            try:
                def wght(X): return weight(X, scale=scale)
                psi = ShiftTester(wght, T, lambda n: n**0.4, replacement="NO-REPL-reject", reject_retries=500)
                out.append(psi.test(X))
            except:
                # Catch errors from test statistic
                print(f"Error occurred {repl}, {scale}, {n}")
                out.append(np.nan)
    return(out)

## Conduct multiple experiments with multiprocessing and export to R for plotting:
if __name__ == '__main__':
    repeats = 10000
    pool = Pool(cpu_count()-2)
    res = zip(*list(tqdm(pool.imap_unordered(conduct_experiment, range(repeats)), total=repeats)))
    pool.close()

    res = (np.array(res) < 0.05).mean(axis=0)
    df = pd.DataFrame(
        [(x, conf, *v) for x, conf, v in zip(res, list(product(causal_effects, n_range, scales, replace_range)))],
        columns=["alpha", "CausalEffect", "n", "Scale", "Replacement"])

    # Export to R for ggplotting
    df['alpha'] = df["alpha"].replace(np.NaN, "NA")
    df.to_csv("experiments/section-5-1/choice-sigma.csv")
