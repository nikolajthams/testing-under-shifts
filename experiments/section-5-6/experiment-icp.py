import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from collections import Counter

import os
import sys
try: sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError: print("Cannot load testing module")
from resample_and_test import ShiftTester

# Load ICP into R
import rpy2.robjects as robj
import rpy2.robjects.packages as packages
import rpy2.robjects.numpy2ri
InvariantCausalPrediction = packages.importr("InvariantCausalPrediction")
rpy2.robjects.numpy2ri.activate()
def to_r(x): return robj.FloatVector(x)

np.random.seed(1)

# Help functions
def e(n=1): return np.random.normal(size=(n, 1))  # Gaussian noise
def cb(*args): return np.concatenate(args, axis=1)  # Col bind
def rb(*args): return np.concatenate(args, axis=0)  # Row bind
p = norm.pdf

# Simulate data from a Gaussian SCM
# Setting == 1 is the one presented in a table, setting 2 is the graph discussed at the very end of the section
sigma_obs = 2
sigma_target = 1

def get_lik(X): return p(X[:,2], loc = X[:,1] + X[:,3], scale=sigma_obs)
def scm(n, setting=1):
    if setting == 1:
        X1 = e(n)
        X3 = e(n)
        X2 = X3 + X1 + sigma_obs*e(n)
        Y = X2 + X3 + 0.3*e(n)
        X4 = -1*Y + X2 + X3 + 0.7*e(n)
    else:
        X4 = e(n)
        Y = X4 + e(n)
        X1 = Y + e(n)
        X3 = e(n)
        X2 = X1 + 0.2*X3 + sigma_obs*e(n)
    return cb(Y, X1, X2, X3, X4)

# Simulate data from a Gaussian SCM
def target_scm(n, setting=1):
    if setting == 1:
        X1 = e(n)
        X3 = e(n)
        X2 = -X3 + X1 + sigma_target*e(n)
        Y = X2 + X3 + 0.3*e(n)
        X4 = -1*Y + X2 + X3 + 0.7*e(n)
    else:
        X4 = e(n)
        Y = X4 + e(n)
        X1 = Y + e(n)
        X3 = e(n)
        X2 = X1 - 0.2*X3 + sigma_target*e(n)

    out = cb(Y, X1, X2, X3, X4)
    return out

def get_weight(setting = 1):
    def weight(X):
        if setting == 1:
            num = p(X[:, 2], loc=X[:,1]-X[:,3], scale=sigma_target)
            denom = p(X[:, 2], loc=X[:,1] + X[:,3], scale=sigma_obs)
        else:
            num = p(X[:, 2], loc=X[:,1]-0.2*X[:,3], scale=sigma_target)
            denom = p(X[:, 2], loc=X[:,1] + 0.2*X[:,3], scale=sigma_obs)
        return num/denom
    return weight

n = 1000
m = 30
vars = np.array([1, 2, 3, 4])

outs = {}

settings = [1]
n_repeats = 500

for setting in settings:
    outs[(setting, "Resampled")] = []
    outs[(setting, "Target")] = []
    psi = ShiftTester(get_weight(setting), T=None, rate=None, replacement="NO-REPL-reject", reject_retries=500)

    for _ in tqdm(range(n_repeats)):
        e1 = scm(n, setting)
        e2 = target_scm(m, setting)
        data = rb(e1, e2)

        # Test with sampled target
        Y, X = data[:,0], data[:,1:]
        Y = to_r(Y)
        nr,nc = X.shape
        X = robj.r.matrix(X, nrow=nr, ncol=nc)
        IN = robj.r.list(to_r(np.arange(1, e1.shape[0]+1)), to_r(np.arange(e1.shape[0]+1, data.shape[0]+1)))
        p_vals = InvariantCausalPrediction.ICP(X=X, Y=Y, ExpInd=IN, test="exact", maxNoObs=5000, alpha=0.05, showAcceptedSets=False, showCompletion=False).rx2['pvalues']
        outs[(setting, 'Target')].append(tuple(vars[list(p_vals < 0.05)]))

        # Test with resampled data
        e3 = psi.resample(e1, m=m)
        data = rb(e1, e3)

        # Test with sampled target
        Y, X = data[:,0], data[:,1:]
        Y = to_r(Y)
        nr,nc = X.shape
        X = robj.r.matrix(X, nrow=nr, ncol=nc)
        IN = robj.r.list(to_r(np.arange(1, e1.shape[0]+1)), to_r(np.arange(e1.shape[0]+1, data.shape[0]+1)))
        p_vals = InvariantCausalPrediction.ICP(X=X, Y=Y, ExpInd=IN, test="exact", maxNoObs=5000, alpha=0.05, showAcceptedSets=False, showCompletion=False).rx2['pvalues']
        outs[(setting, 'Resampled')].append(tuple(vars[list(p_vals < 0.05)]))


found_sets = np.unique([x for v in outs.values() for x in v])

final_tables={}
for setting in settings:
    for data_name in ("Resampled", "Target"):
        x = Counter(outs[(setting, data_name)])
        print(f"{data_name}, setting {setting}")
        for s, v in x.items(): print(str(s).ljust(20) + str(v))
        final_tables[(setting, data_name)] = x

# Convert to LaTeX table
out = "Set & Frequency \\\\ \\hline \n"
for s in found_sets:
    if len(s) == 0:
        out += "$\emptyset$ &"
    else:
        out += "$\{" + ",".join(str(x) for x in s) + "\}$ &"
    for setting in settings:
        out += f"${np.round(final_tables[(setting, 'Resampled')][s]/n_repeats*100, 1)}\%"
        out += f" ({np.round(final_tables[(setting, 'Target')][s]/n_repeats*100, 1)}\%)$" #+ ("&" if setting == 1 else "")
    out += "\\\\ \n"

with open("experiments/section-5-6/icp-table.tex", "w") as text_file:
    text_file.write(out)