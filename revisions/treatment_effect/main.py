import numpy as np
import pandas as pd
import argparse
from itertools import product
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import ttest_ind
import statsmodels.api as sm
from scipy.stats import norm
from resample_and_test import ShiftTester
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from doubleml import DoubleMLPLR, DoubleMLData
from datetime import datetime
from multiprocessing import Pool, cpu_count
import os
from tqdm import tqdm
import json
from functools import partial


p = norm.pdf

argparser = argparse.ArgumentParser()
argparser.add_argument('--n_range', type=int, nargs='+', default=[int(10**p) for p in [2, 2.5, 3, 3.5, 4]])
argparser.add_argument('--seed', type=int, default=None)
argparser.add_argument('--z_dim', type=int, default=20)
argparser.add_argument('--causal_effect', type=float, nargs='+', default=[0])
argparser.add_argument('--zy_complexity', type=float, default=1)
argparser.add_argument('--reweight_target', type=str, default="X")
argparser.add_argument('--reweight_covariate', type=str, default=[f"Z_{i}" for i in range(20)])
argparser.add_argument('--test_target', type=str, default="Y")
argparser.add_argument('--test_covariate', type=str, default="X")
argparser.add_argument('--test_confounder', type=str, default=[f"Z_{i}" for i in range(20)])
argparser.add_argument('--replacement', type=str, default='False')
argparser.add_argument('--n_cpu', type=int, default=cpu_count()-2)
argparser.add_argument('--main_dir', type=str, default="revisions/treatment_effect")
argparser.add_argument('--n_repeats', type=int, default=100)
argparser.add_argument('--alpha', type=float, default=0.05)

def cb(*args):
    return np.concatenate(args, axis=1)

def e(n=1, mu=0, dim=1): 
    return np.random.normal(size=(n, dim), loc=mu)  # Gaussian noise

def coin(n, p=0.5):
    return 1*(np.random.uniform(size=(n,1)) < p)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def scm(n, causal_effect, args, seed=None):
    if seed is not None:
        np.random.seed(seed)
    Z = e(n, dim=args.z_dim)
    Z_sum = np.sum(Z, axis=1, keepdims=True)
    X = coin(n, p=sigmoid(Z_sum))
    Y = causal_effect * X + np.exp(-Z_sum**2/2)*np.sin(args.zy_complexity*Z_sum) + np.sqrt(0.1)*e(n)
    
    # Create data frame with X, Y and Z
    df = pd.DataFrame({"X": X.reshape(-1), "Y": Y.reshape(-1)} | {f"Z_{i}": Z[:,i] for i in range(args.z_dim)})

    return df

def get_weight(args):
    reweight_target = args.reweight_target
    reweight_covariate = args.reweight_covariate
    def weight_fitted(data):
        # Regress to learn old mean and scale
        learner_treatment = LogisticRegression()
        learner_treatment.fit(data[reweight_covariate], data[reweight_target])
        q = learner_treatment.predict_proba(data[reweight_covariate])
        q = data[reweight_target]*q[:,1] + (1-data[reweight_target])*q[:,0]

        # Set target mean and scale as marginal distribution
        marginal_p = data[reweight_target].mean()
        target_p = data[reweight_target]*marginal_p + (1-data[reweight_target])*(1-marginal_p)

        return target_p / q
    return weight_fitted

def get_p_val(args):
    test_target = args.test_target
    test_covariate = args.test_covariate
    def p_val(X):
        # Check that both groups are present
        if np.sum(X[test_covariate]) <= 1 or np.sum(X[test_covariate]) >= (len(X[test_covariate]) - 1):
            return 0.5

        return ttest_ind(X[X[test_covariate] == 1][test_target], X[X[test_covariate] == 0][test_target]).pvalue
    return p_val

def get_double_ml_binary_treatment(args):
    def double_ml_binary_treatment(data):
        # Initiate DML data
        dml_data = DoubleMLData(data,
                                y_col=args.test_target,
                                d_cols=args.test_covariate,
                                x_cols=args.test_confounder)
        # Initiate DML models
        learner_outcome = RandomForestRegressor(n_estimators = 100, max_features = 'sqrt', max_depth= 3)
        learner_treatment = LogisticRegression()

        # Fit DML models and get p-value
        dml_model = DoubleMLPLR(dml_data, ml_l = learner_outcome, ml_m = learner_treatment)
        dml_model.fit()
        return dml_model.pval[0]
    return double_ml_binary_treatment

def experiment(seed=None, args=None):
    # Initiate double_ml model
    double_ml_binary_treatment = get_double_ml_binary_treatment(args)
    out = []

    for n, ce in product(args.n_range, args.causal_effect):
        data = scm(n=n, causal_effect=ce, args=args, seed=seed)
        psi = ShiftTester(weight=get_weight(args), p_val = get_p_val(args), degenerate="retry")
        # Add ours
        out.append({
            "method": "ours",
            "reject": 1.0*(psi.combination_test(data, replacement=args.replacement, m=int(np.sqrt(len(data))), method="cct") < args.alpha),
            "n": n,
            "causal_effect": ce
        })
        # Add double_ml
        out.append({
            "method": "DML",
            "reject": 1.0*(double_ml_binary_treatment(data) < args.alpha),
            "n": n,
            "causal_effect": ce
        })
    return out

if __name__ == "__main__":
    args = argparser.parse_args()
    time_string = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    MAIN_DIR = args.main_dir

    # Define partial experiment function which takes args as input
    experiment_partial = partial(experiment, args=args)

    # Multiprocess
    pool = Pool(args.n_cpu)
    res = np.array(
        list(tqdm(pool.imap_unordered(experiment_partial, range(args.n_repeats)), total=args.n_repeats)))
    pool.close()
    res = [item for sublist in res for item in sublist]
        
    # Check if relevant folders exists
    for folder in ["results", "args"]:
        if not os.path.exists(os.path.join(MAIN_DIR, folder)):
            os.makedirs(os.path.join(MAIN_DIR, folder))
    
    # Save data
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(MAIN_DIR, "results", f"{time_string}.csv"), index=False)
    df.to_csv(os.path.join(MAIN_DIR, "latest.csv"), index=False)

    # Save args file
    with open(os.path.join(MAIN_DIR, "args", f"{time_string}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    def get_ci(rejects):
        lower, upper = proportion_confint(count=rejects.sum(), nobs = len(rejects), alpha=0.05, method="wilson")
        return lower, rejects.mean(), upper
    
    df = df.groupby([i for i in df.columns if i != "reject"]).aggregate(get_ci)
    for n,col in enumerate(["lower", "mean", "upper"]):
        df[col] = df['reject'].apply(lambda x: x[n])
    df = df.drop('reject',axis=1).rename({"mean": "reject"}, axis=1).reset_index()

    df.to_csv(os.path.join(MAIN_DIR, "results", f"{time_string}.csv"), index=False)
    df.to_csv(os.path.join(MAIN_DIR, "latest.csv"), index=False)
