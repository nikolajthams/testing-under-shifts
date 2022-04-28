import numpy as np
from scipy.stats import norm
from multiprocessing import Pool, cpu_count
import statsmodels.api as sm
from tqdm import tqdm
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


def experiment(i=1):
    psi = ShiftTester(weight, T, replacement=False, verbose=False)
    res = []
    for c_e in [0, 0.4]:
        X = scm(n, causal_effect=c_e)
        for m in ([10**(x/4) for x in range(2, 17)]):
            out = []
            for rep in range(100): out.append(psi.test(X, m=int(m)))
            res.append([n, m, c_e, np.mean(out)])
    return res


if __name__ == '__main__':
    res = []
    test_output = []
    for c_e in [0, 0.4]:
        psi = ShiftTester(weight, T, replacement=False, verbose=False)
        for idx in range(1):
            X = scm(n, causal_effect=c_e)
            for m in tqdm([10**(x/4) for x in range(2, 17)]):
                for _ in range(100):
                    try:
                        resamp = psi.resample(X, m=int(m))
                        pval = psi.gaussian_validity(resamp, cond=[0], j_x=[0], j_y=[1], return_p=True)
                        res.append([idx, _, n, m, pval, c_e, psi.T(resamp)])
                    except:
                        res.append([idx, _, n, m, np.nan, c_e, np.nan])

    df = pd.DataFrame(res, columns=["idx", "rep", "n", "m", "pval", "causal_effect", "hyp_test"])
    df['pval'] = df["pval"].replace(np.NaN, "NA")
    df['hyp_test'] = df["hyp_test"].replace(np.NaN, "NA")

    # Export to R for ggplotting
    df.to_csv("experiments/section-5-1/choice-m.csv")

    repeats = 500
    pool = Pool(cpu_count()-1)
    res = np.concatenate(
        list(tqdm(pool.imap_unordered(experiment, range(repeats)), total=repeats)), axis=0)
    pool.close()

    df = pd.DataFrame(res, columns=["n", "m", "causal_effect", "hyp_test"])
    df['hyp_test'] = df["hyp_test"].replace(np.NaN, "NA")

    # Export to R for ggplotting
    df.to_csv("experiments/section-5-1/choice-m-ci.csv")

