import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import json
from scipy.stats import norm, cauchy
from resample_and_test import ShiftTester
import statsmodels.api as sm
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import product
from tqdm import tqdm
from statsmodels.gam.api import GLMGam, BSplines
from datetime import datetime
import argparse
argparser = argparse.ArgumentParser()

# Output folders
MAIN_DIR = "revisions/combination_test"

# Helper functions
def e(n=1, mu=0): 
    return np.random.normal(size=(n, 1), loc=mu)  # Gaussian noise
def coin(n, p=0.5):
    return 1*(np.random.uniform(size=(n,1)) < 0.5)
def cb(*args): 
    return np.concatenate(args, axis=1)  # Col bind
p = norm.pdf

# Simulate data from a linear SCM
def linear_scm(n, causal_effect=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    C = e(n) # Bimodal Gaussian distribution
    X = 3*C + 2*e(n)
    Y = C + causal_effect*X + 2*e(n)
    return cb(C, X, Y)

# Simulate data from a non-linear SCM
def scm(n, causal_effect=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X1 = e(n, mu=4*coin(n)-2) # Bimodal Gaussian distribution
    X2 = -X1**2 + 2*e(n)
    X3 = np.sin(X2) + causal_effect*X1 + 2*e(n)
    return cb(X1, X2, X3)

# Function for getting raw p-values
def p_val(X):
    return sm.OLS(X[:, 2], sm.tools.add_constant(X[:, 0])).fit().pvalues[1]

# Function for getting test statistic.
def get_T(causal_effect=0, alpha=0.05, test_CI=False, target=2, covariate=0):
    # Test hypothesis of 0 correlation
    def T(X):
        return 1.0*(sm.OLS(X[:, target], sm.tools.add_constant(X[:, covariate])).fit().pvalues[1]<alpha)
    # Test hypothesis that regression slop is the specified causal effect
    def T_CI(X):
        lower, upper = sm.OLS(X[:, target], sm.tools.add_constant(X[:, covariate])).fit().conf_int(alpha=alpha)[1]
        return 1.0*((lower > causal_effect) | (causal_effect > upper))
    return T_CI if test_CI else T


# Weights
def weight_fitted(X):
    # Regress to learn old mean and scale
    results = GLMGam(X[:,1], smoother=BSplines(X[:,0], df = 10, degree=3)).fit()
    old_mean = results.fittedvalues
    old_scale = np.sqrt(results.scale)

    # Set target mean and scale as marginal distribution
    new_mean = X[:,1].mean()
    new_scale = np.sqrt(X[:,1].var())
    return p(X[:,1], loc=new_mean, scale=new_scale)/p(X[:,1], loc=old_mean, scale=old_scale)

def linear_weight_fitted(X):
    # Regress to learn old mean and scale
    results = sm.OLS(X[:,1], sm.tools.add_constant(X[:,0])).fit()
    old_mean = results.fittedvalues
    old_scale = np.sqrt(results.scale)

    # Set target mean and scale as marginal distribution
    new_mean = X[:,1].mean()
    new_scale = np.sqrt(X[:,1].var())

    return p(X[:,1], loc=new_mean, scale=new_scale)/p(X[:,1], loc=old_mean, scale=old_scale)


# Add arguments
argparser.add_argument('--m_rate', type=float, default=0.5)
argparser.add_argument('--seed', type=int, default=None)
argparser.add_argument('--n_combinations', type=int, default=100)
argparser.add_argument('--replacement', type=str, default='False')
argparser.add_argument('--n_processes', type=int, default=cpu_count())
argparser.add_argument('--alpha', type=float, default=0.05)
argparser.add_argument('--effect_vanish_power', type=float, default=0)
argparser.add_argument('--causal_effect_multiplier', type=float, nargs='+', default=[0])
argparser.add_argument('--n_repeats', type=int, default=1000)
argparser.add_argument('--n_range', type=int, nargs='+', default=[int(10**p) for p in [2, 2.5, 3, 3.5, 4]])
argparser.add_argument('--n_cpu', type=int, default=cpu_count()-2)
argparser.add_argument('--test_CI', type=bool, default=False)
argparser.add_argument('--target', type=int, default=2)
argparser.add_argument('--covariate', type=int, default=0)
argparser.add_argument('--include_heuristic', type=bool, default=True)
argparser.add_argument('--quantile_repeats', type=int, default=10)


def experiment(seed=None, args=None):
    out = []
    for n, ce_multiplier in product(args.n_range, args.causal_effect_multiplier):
        causal_effect = ce_multiplier*(n**args.effect_vanish_power)
        T = get_T(causal_effect=causal_effect, alpha=args.alpha, test_CI=args.test_CI, target=args.target, covariate=args.covariate)
        data = scm(n, causal_effect=causal_effect, seed=seed)
        psi = ShiftTester(weight_fitted, T=T, replacement=args.replacement, p_val=p_val)
        m = int(n**args.m_rate)
        # Conduct single test
        out.append({
                "method": "single-test",
                "reject": psi.test(data, m=m),
                "n": n,
                "causal_effect": causal_effect
            })
        # Conduct combination tests
        for combination_type in ["hartung", "meinshausen", "cct"]:
            out.append({
                    "method": combination_type,
                    "reject": psi.combination_test(data, m=m, n_combinations=args.n_combinations, method=combination_type, warn=False, alpha=args.alpha),
                    "n": n,
                    "causal_effect": causal_effect
                })
        # Conduct single test with heuristic
        if args.include_heuristic:
            p_cutoff = np.quantile(np.random.uniform(size=(1000, args.quantile_repeats)).min(axis=1), 0.05)
            m_heuristic = psi.tune_m(data, j_x=[0], j_y=[1], gaussian=True, repeats=args.quantile_repeats, cond=[0], m_init=int(np.sqrt(n)), p_cutoff=p_cutoff, m_factor=1.3)
            out.append({
                    "method": "heuristic",
                    "reject": psi.test(data, m=m_heuristic),
                    "n": n,
                    "causal_effect": causal_effect
                })
    return out

if __name__ == "__main__":
    args = argparser.parse_args()
    time_string = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    # Define partial experiment function which takes args as input
    experiment_partial = partial(experiment, args=args)

    # Multiprocess
    pool = Pool(args.n_cpu)
    res = np.array(
        list(tqdm(pool.imap_unordered(experiment_partial, range(args.n_repeats)), total=args.n_repeats)))
    pool.close()
        
    # Check if relevant folders exists
    for folder in ["results", "args"]:
        if not os.path.exists(os.path.join(MAIN_DIR, folder)):
            os.makedirs(os.path.join(MAIN_DIR, folder))
    
    # Save data
    res = [item for sublist in res for item in sublist]
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(MAIN_DIR, "results", f"{time_string}.csv"), index=False)
    df.to_csv(os.path.join(MAIN_DIR, "latest.csv"), index=False)

    # Save args file
    with open(os.path.join(MAIN_DIR, "args", f"{time_string}.json"), "w") as f:
        json.dump(vars(args), f)