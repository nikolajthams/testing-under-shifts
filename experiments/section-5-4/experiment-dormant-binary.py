from itertools import product
import numpy as np
from multiprocessing import Pool, cpu_count
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
import pandas as pd
from scipy.stats import fisher_exact
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro

from resample_and_test import ShiftTester

base_r = ro.r
base_r['source']('nested_markov_test.R')


def pandas_to_r(pd_df):
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(pd_df)
    return r_from_pd_df


def gen_params(random_state, dim_u=4, null=True):
    p_x1 = random_state.beta(1, 1)
    p_u = random_state.dirichlet(np.ones(shape=(dim_u,)) * 3)
    p_x2 = random_state.beta(1, 1, size=(2, dim_u))
    p_x3 = random_state.beta(1, 1, size=(2,))
    if null:
        p_x4 = random_state.beta(1, 1, size=(2, dim_u))
    else:
        p_x4 = random_state.beta(1, 1, size=(2, 2, dim_u))

    return p_x1, p_u, p_x2, p_x3, p_x4


def gen_data(n, params, random_state, null=True):
    p_x1, p_u, p_x2, p_x3, p_x4 = params
    dim_u = p_u.shape[0]
    U = random_state.choice(dim_u, size=n, p=p_u)
    X1 = random_state.binomial(1, p_x1, size=n)
    X2 = random_state.binomial(1, p_x2[X1, U], size=(1, n)).flatten()
    X3 = random_state.binomial(1, p_x3[X2], size=(1, n)).flatten()
    if null:
        X4 = random_state.binomial(1, p_x4[X3, U], size=(1, n)).flatten()
    else:
        X4 = random_state.binomial(1, p_x4[X1, X3, U], size=(1, n)).flatten()

    df = pd.DataFrame(np.vstack([X1, X2, X3, X4]).T, columns=['x1', 'x2', 'x3', 'x4'])

    return df


def get_freq(df):
    df['freq'] = 1
    df = df.groupby(['x1', 'x2', 'x3', 'x4'])['freq'].sum().reset_index()
    return df


def weight_fit(X, p=None):
    if p is None:
        p = X['x3'].mean()
    p_X3 = np.array([1 - p, p])
    X2X3 = get_freq(X).groupby(['x2', 'x3'])['freq'].sum().reset_index()
    X2X3['prob'] = X2X3.groupby('x2')['freq'].transform(lambda x: x / np.sum(x))

    weights = pd.merge(X, X2X3, on=['x2', 'x3'], how='left')['prob'].to_numpy()
    weights = p_X3[X['x3']] / weights

    return weights


def optimize_weight(X, tester):
    grid = np.linspace(0.05, 0.95, 10)
    res = []
    for p in grid:
        rejection_rate = []
        for _ in range(50):
            p_X3 = np.array([1 - p, p])
            X2X3 = get_freq(X).groupby(['x2', 'x3'])['freq'].sum().reset_index()
            X2X3['prob'] = X2X3.groupby('x2')['freq'].transform(lambda x: x / np.sum(x))

            weights = pd.merge(X, X2X3, on=['x2', 'x3'], how='left')['prob'].to_numpy()
            weights = p_X3[X['x3']] / weights

            psi = tester(weights)

            rejection_rate += [psi.test(X)]
        res += [rejection_rate]

    max_p = np.asarray(res).mean(axis=1).argmax()

    return grid[max_p]


# Define rates for resampling
def rate(pow, c=1):
    def f(n): return c * n ** pow

    return f


def cross_tab(x, col1, col2):
    cros_df = pd.crosstab(x[col1], x[col2])
    nrow, ncol = cros_df.shape
    if nrow < 2:
        new_idx = np.setdiff1d(np.array([0, 1]), cros_df.index.values)[0]
        new_val = pd.Series([0, 0], name=new_idx)
        cros_df = cros_df.append(new_val)
    if ncol < 2:
        new_idx = np.setdiff1d(np.array([0, 1]), cros_df.columns.values)[0]
        cros_df[new_idx] = 0

    return cros_df


def fisher_test(X):
    X1X4 = cross_tab(X, 'x1', 'x4')
    _, p_val = fisher_exact(X1X4)

    return p_val < 0.05


def ll_ratio_test(X):
    r_df = pandas_to_r(get_freq(X))
    p_val = base_r.nmm_test(r_df)[0]

    return p_val < 0.05


mode = 'varopt'  # 'varopt', 'pvalopt', unif

pow = 0.5
n_range = [100, 300, 900, 2700, 8100, 24300]
tests = ["Resampling test", "Score-based test"]
hypothesize = ['null', 'alternative']
combinations = list(product(n_range))


def fit_m(X, p=None, n_iter=10, threshold=0.29):
    n = X.shape[0]
    m_size = np.linspace(n ** 0.5, n ** 0.9, 12)

    wf = lambda x: weight_fit(x, p=p)
    ret_m = m_size[0]

    for m in m_size:
        psi_alter = ShiftTester(wf, fisher_test, rate(1, c=m / n),
                                  replacement=False)
        reX = [psi_alter.resample(X) for _ in range(n_iter)]
        pval = [fisher_exact(cross_tab(x, 'x3', 'x2'))[1] for x in reX]

        if np.mean(pval) > threshold:
            ret_m = m
        else:
            break

    return ret_m


## Wrap as function to apply multiprocessing
def conduct_experiment(i=None):
    out = []
    param_state = lambda: np.random.RandomState(i)
    param_null = gen_params(param_state(), dim_u=4, null=True)
    param_alter = gen_params(param_state(), dim_u=4, null=False)

    for n in n_range:
        random_state = np.random.RandomState((i, n))
        X_null = gen_data(n, param_null, random_state, null=True)
        X_alter = gen_data(n, param_alter, random_state, null=False)

        try:
            # lltest
            lltest_null = False
            lltest_alter = False

            # optimize m
            m_null = fit_m(X_null, p=None)
            rate_null = rate(1, c=m_null / n)
            m_alter = fit_m(X_alter, p=None)
            rate_alter = rate(1, c=m_alter / n)

            if mode == 'pvalopt':
                # optimize weight
                tester = lambda w: ShiftTester(w, fisher_test, rate(pow, c=1), replacement=True)
                optimized_p_null = optimize_weight(X_null, tester)
                optimized_p_alter = optimize_weight(X_alter, tester)
                wf_null = lambda x: weight_fit(x, optimized_p_null)
                wf_alter = lambda x: weight_fit(x, optimized_p_alter)
                pass
            elif mode == 'varopt':
                wf_null = weight_fit
                wf_alter = weight_fit
            elif mode == 'unif':
                wf_null = lambda x: weight_fit(x, .5)
                wf_alter = lambda x: weight_fit(x, .5)
            else:
                raise Exception('mode is not defined')

            psi_null = ShiftTester(wf_null, fisher_test,
                                     rate_null, replacement=False)
            psi_alter = ShiftTester(wf_alter, fisher_test,
                                      rate_alter, replacement=False)

            out.append((psi_null.test(X_null), lltest_null,
                        psi_alter.test(X_alter), lltest_alter))

        except Exception as e:
            # Catch errors from test statistic
            print(f"Error occurred {n}")
            print(e)
            out.append((np.nan, np.nan, np.nan, np.nan))
    return out


## Conduct multiple experiments with multiprocessing and export to R for plotting:
if __name__ == '__main__':
    repeats = 500
    # Multiprocess
    pool = Pool(cpu_count() - 2)
    res = np.array([ret for ret in tqdm(pool.imap_unordered(conduct_experiment, range(repeats)),
                                        total=repeats) if ret is not None])
    pool.close()

    alpha_ipr_null = res[:, :, 0]
    alpha_score_null = res[:, :, 1]
    alpha_ipr_alter = res[:, :, 2]
    alpha_score_alter = res[:, :, 3]


    def get_df_alpha(alpha, test, hypothesis):
        # # Count non-nas, to be used for binomial confidence intervals
        counts = (~np.isnan(alpha)).sum(axis=0)
        alpha = np.nansum(alpha, axis=0)
        # Pack as data frame
        df_alpha = pd.DataFrame(
            [(x / c, *v, *proportion_confint(x, c, method='binom_test')) for x, v, c in
             list(zip(alpha, combinations, counts))],
            columns=["alpha", "n", "Lower", "Upper"])

        df_alpha['Test'] = test
        df_alpha['Hypothesis'] = hypothesis

        return df_alpha


    df_ipr_null = get_df_alpha(alpha_ipr_null, 'Resampling Test', 'null')
    df_score_null = get_df_alpha(alpha_score_null, 'Score-based Test', 'null')
    df_ipr_alter = get_df_alpha(alpha_ipr_alter, 'Resampling Test', 'alternative')
    df_score_alter = get_df_alpha(alpha_score_alter, 'Score-based Test', 'alternative')

    df = pd.concat([df_ipr_null, df_score_null, df_ipr_alter, df_score_alter], ignore_index=True)

    # Export to R for ggplotting
    df['alpha'] = df["alpha"].replace(np.NaN, "NA")
    df.to_csv("experiments/section-5-4/experiment-dormant-binary.csv")
