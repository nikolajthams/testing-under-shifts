import numpy as np
from scipy.stats import norm
from functools import partial
from resample_and_test import ShiftTester
from itertools import product
import statsmodels.api as sm
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import argparse

argparser = argparse.ArgumentParser()


# Helper functions
def e(n=1, mu=0): 
    return np.random.normal(size=(n, 1), loc=mu)  # Gaussian noise
def coin(n, p=0.5):
    return 1*(np.random.uniform(size=(n,1)) < 0.5)
def cb(*args): 
    return np.concatenate(args, axis=1)  # Col bind
p = norm.pdf


# Simulate data from a Gaussian SCM
def scm(n, causal_effect=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    C = e(n) # Bimodal Gaussian distribution
    X = 3*C + 2*e(n)
    Y = C + causal_effect*X + 2*e(n)
    return cb(C, X, Y)

X = scm(1000)
results = sm.OLS(X[:,1], sm.tools.add_constant(X[:, 0])).fit()
results.summary()
results.scale

# Weight
def weight(X):
    old_mean = 3*X[:,0]
    old_scale = 2

    new_mean = X[:,1].mean()
    new_scale = np.sqrt(X[:,1].var())
    return p(X[:,1], loc=new_mean, scale=new_scale)/p(X[:,1], loc=old_mean, scale=old_scale)

# Weight
def weight_fitted(X):
    results = sm.OLS(X[:,1], sm.tools.add_constant(X[:,0])).fit()
    old_mean = results.predict(sm.tools.add_constant(X[:,0]))
    old_scale = np.sqrt(results.scale)

    new_mean = X[:,1].mean()
    new_scale = np.sqrt(X[:,1].var())
    return p(X[:,1], loc=new_mean, scale=new_scale)/p(X[:,1], loc=old_mean, scale=old_scale)



# Test function: LinReg
def T(X):
    return (sm.OLS(X[:, 2], sm.tools.add_constant(X[:, 1])).fit().pvalues[1] < 0.05)*1

def get_T(causal_effect):
    def T_(X):
        lower, upper = sm.OLS(X[:, 2], sm.tools.add_constant(X[:, 1])).fit().conf_int()[1]
        return (lower > causal_effect) | (causal_effect > upper)
    return T_



# Compute for goodness-of-fit quantile
quantile_repeats = 15
cutoff = np.quantile(np.random.uniform(size=(1000, quantile_repeats)).min(axis=1), 0.005)

argparser.add_argument("--n_range", type=int, default=[10, 100, 1000, 10000])
argparser.add_argument("--power_range", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
argparser.add_argument("--repeats", type=int, default=1000)
argparser.add_argument("--causal_effect_multiplier", type=float, default=1.0)
argparser.add_argument("--m_power", type=float, default=0.5)

def experiment(seed=None, args=None):
    out = []
    for power, n in product(args.power_range, args.n_range):
        causal_effect = args.causal_effect_multiplier*(n**power)
        data = scm(n, causal_effect=causal_effect, seed=seed)
        psi = ShiftTester(weight_fitted, get_T(causal_effect), rate=None, replacement=False)
        # m = psi.tune_m(data, j_x=[0], j_y=[1], gaussian=True, repeats=quantile_repeats, cond=[0], m_init=int(np.sqrt(n)), p_cutoff=cutoff, m_factor=1.3)
        m = int(n**args.m_power)
        out.append(
            {
                "reject": psi.test(data, m=m),
                "n": n,
                "power": power
            }
            )
    return out


if __name__ == '__main__':
    args = argparser.parse_args()

    # Define partial experiment function which takes args as input
    experiment_partial = partial(experiment, args=args)
    
    # Multiprocess
    pool = Pool(cpu_count()-2)
    res = np.array(
        list(tqdm(pool.imap_unordered(experiment, range(args.repeats)), total=args.repeats)))
    pool.close()
    res = [item for sublist in res for item in sublist]
    df = pd.DataFrame(res)
    df.to_csv("experiment-parametric-rate/experiment-parametric-rate.csv", index=False)
