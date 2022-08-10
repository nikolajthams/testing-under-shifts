import os
import matplotlib.pyplot as plt
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from resample_and_test import ShiftTester

from revisions.combination_test.weights_and_tests import get_weight_func
from revisions.combination_test.data_generation import e, cb

# Work in progress non-linear SCM
def scm(n, causal_effect=0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = 2*e(n)
    C = np.abs(X-1) + 4*e(n)
    Y = C + causal_effect*X + 2*e(n)
    return cb(X, C, Y)



n = 1000
data = scm(n=n, causal_effect=1)
psi = ShiftTester(get_weight_func(), T=None, replacement=False, p_val=None)
data_resamp = psi.resample(data, m=int(n**0.5))
var_names = ['X', 'C', 'Y']
p_val_obs = sm.OLS(data[:,2], sm.tools.add_constant(data[:,0])).fit().pvalues[1]
p_val_resamp = sm.OLS(data_resamp[:,2], sm.tools.add_constant(data_resamp[:,0])).fit().pvalues[1]

# Make plot with three subplots. Plot the three pairwise conditionals of data
fig, axs = plt.subplots(3, 3, figsize=(15,15))
for i, d in enumerate([data, data_resamp]):
    for j, pair in enumerate([(0,1), (0,2), (1,2)]):
        axs[i, j].scatter(d[:,pair[0]], d[:,pair[1]])
        axs[i, j].set_xlabel(var_names[pair[0]])
        axs[i, j].set_ylabel(var_names[pair[1]])
        axs[i, j].set_title(f'{var_names[pair[0]]} vs {var_names[pair[1]]}')
model = GLMGam(data[:,1], smoother=BSplines(data[:,0], df = 10, degree=3, include_intercept=True)).fit()
x_range = np.linspace(data[:,0].min(), data[:,0].max(), 200)
def f(x):
    return np.abs(x-1)
axs[0, 0].plot(x_range, model.predict(exog_smooth=x_range, transform=True), color='red')
axs[0, 0].plot(x_range, f(x_range), color='green')

axs[2, 0].scatter(data[:,0], data[:,1] - model.predict(exog_smooth=data[:,0], transform=True), color='red')

# Save figure
fig.text(0.5, 0.01, f'Observed p-value {p_val_obs:.4}. Resampled p-value {p_val_resamp:.4}', ha='center', va='center', fontsize=20)
fig.savefig('scm.png')

