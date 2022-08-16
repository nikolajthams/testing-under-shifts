import numpy as np
import pandas as pd
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as packages
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

from KernelLogrankTest.kernel_logrank.tests import wild_bootstrap_LR
from resample_and_test import ShiftTester

rpy2.robjects.numpy2ri.activate()
dHSIC = packages.importr("dHSIC")

# Set some parameters
B = 1000
np.random.seed(1)
# Load data
colon = pd.read_csv('colon.csv')
colon = colon[colon.etype == 2]
covariates = ['differ', 'node4', 'age', 'perfor', 'sex', 'obstruct', 'adhere', 'surg', 'extent', 'rx']
full_data = colon[covariates + ['time', 'status']]

print('percentage observed', full_data.status.mean())

# fit logistic regression model
y_col = 'time'
x_col = 'obstruct'
adjust = ['age', 'sex', 'extent']
formula = "{} ~ {}".format(x_col, '+'.join(adjust))
glm_model = smf.glm(formula=formula, data=full_data.dropna(), family=sm.families.Binomial()).fit()
print(glm_model.summary())
roc_auc_score(full_data.dropna()[x_col], glm_model.predict())

# computing p-values of the independence tests on the original sample
_, pval_XY = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(X=full_data[x_col].to_numpy()[:, np.newaxis],
                                                                      z=full_data[y_col].to_numpy(),
                                                                      d=full_data['status'].to_numpy(),
                                                                      kernels_x='bin', kernel_z='gau',
                                                                      num_bootstrap_statistics=B)

_, pval_ZY = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(X=full_data[adjust].to_numpy(),
                                                                      z=full_data[y_col].to_numpy(),
                                                                      d=full_data['status'].to_numpy(),
                                                                      kernels_x='gau', kernel_z='gau',
                                                                      num_bootstrap_statistics=B)

pval_XZ = dHSIC.dhsic_test(X=full_data[x_col].to_numpy(), Y=full_data[adjust].to_numpy()).rx2('p.value')[0]


# define p-value function
def p_val(X):
    y = X[:, -1]
    d = X[:, [-2]]
    x = X[:, [0]]
    _, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(X=x, z=y, d=d, kernels_x='bin', kernel_z='gau',
                                                                    num_bootstrap_statistics=B)
    return p


# define test function
def T(X):
    p = p_val(X)
    return 1 * (p < 0.05)


cols = [x_col] + adjust + ['status', 'time']
X = full_data[cols].to_numpy()
target_p1 = full_data[x_col].mean()


# define a weight function
def weight(X):
    df = pd.DataFrame(X, columns=cols)
    prob_1 = glm_model.predict(df).to_numpy()
    prob = prob_1 * (X[:, 0]) + (1 - prob_1) * (1 - X[:, 0])
    target_prob = target_p1 * (X[:, 0]) + (1 - target_p1) * (1 - X[:, 0])
    w = target_prob / prob
    return w


# resampling test
psi = ShiftTester(weight, T, replacement=False, verbose=False, p_val=p_val)

quantile_repeats = 10
cutoff = np.quantile(np.random.uniform(size=(1000, quantile_repeats)).min(axis=1), 0.05)
j_x = [i for i in range(1, len(adjust) + 1)]
j_y = [0]

m = psi.tune_m(X, cond=None, j_x=j_x, j_y=j_y, logistic=True, m_init=200,
               m_factor=1.5,
               repeats=quantile_repeats, p_cutoff=cutoff)

resample_X = psi.resample(X, replacement="REPL-reject", m=m)

# computing p-values of the independence tests on the resample
pval_XZ_resample = dHSIC.dhsic_test(X=resample_X[:, 0], Y=resample_X[:, 1:-2]).rx2('p.value')[0]

pval_ZY_resample = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(X=resample_X[:, 1:-2],
                                                                            z=resample_X[:, -1],
                                                                            d=resample_X[:, -2],
                                                                            kernels_x='gau', kernel_z='gau',
                                                                            num_bootstrap_statistics=B)

pval_XY_resample = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(X=resample_X[:, [0]],
                                                                            z=resample_X[:, -1],
                                                                            d=resample_X[:, -2],
                                                                            kernels_x='bin', kernel_z='gau',
                                                                            num_bootstrap_statistics=B)
