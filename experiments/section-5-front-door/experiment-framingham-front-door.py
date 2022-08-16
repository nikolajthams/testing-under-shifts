import pandas as pd
import numpy as np
from resample_and_test import ShiftTester
import statsmodels.formula.api as smf
import statsmodels.api as sm
import rpy2.robjects.packages as packages
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
# Import rpy for testing with R functions
CondIndTests = packages.importr("CondIndTests")
dHSIC = packages.importr("dHSIC")


def get_data():
    fram_df = pd.read_csv("framingham_data.csv")

    fram_df['HYPER'] = (fram_df['SYSBP'] >= 140) | (fram_df['DIABP'] >= 90)

    mean_BMI = fram_df['BMI'].mean()

    def get_BMI(x):
        if np.isnan(x.iloc[-1]):
            if len(x) > 1:
                return x.iloc[-2]
            else:
                return mean_BMI
        else:
            return x.iloc[-1]

    train_df = fram_df[['RANDID', 'SEX']].drop_duplicates().set_index('RANDID')
    train_df = train_df.join(fram_df.groupby('RANDID')['HYPER'].agg(lambda x: 1 * x.iloc[-1]))
    train_df = train_df.join(fram_df.groupby('RANDID')['PREVCHD'].agg(lambda x: x.iloc[-1]).rename("CHD"))
    train_df = train_df.join(fram_df.groupby('RANDID')['AGE'].agg(lambda x: x.iloc[-1]))
    train_df = train_df.join(fram_df.groupby('RANDID')['BMI'].agg(lambda x: get_BMI(x)))
    train_df = train_df.join(fram_df.groupby('RANDID')['CURSMOKE'].agg(lambda x: x.iloc[-1]).rename("SMOKE"))
    train_df = train_df.join(
        fram_df.groupby('RANDID')['HYPER'].agg(lambda x: 1 * np.any(x.iloc[:-1])).rename("HYPER_HIST"))
    # fillna BMI
    train_df['BMI'] = train_df['BMI'].fillna(mean_BMI)

    return train_df.reset_index(drop=True)


if __name__ == "__main__":

    np.random.seed(0)
    data = get_data()

    A = "SMOKE"  # treatment
    M = "HYPER"  # mediator
    Y = "CHD"  # outcome
    C = ["AGE", "SEX", "BMI"]  # baseline covariates
    Z = "HYPER_HIST"  # anchor

    mpA = C + [Z]
    mpM = C + [Z, A]
    mpY = C + [Z, A, M]


    # defining p-value function
    def p_val(X):
        yvec = X[[Y]].to_numpy()
        zvec = X[[Z]].to_numpy()
        cvec = X[C].to_numpy()

        p = CondIndTests.KCI(yvec, zvec, cvec).rx2('pvalue')[0]

        return p


    def T(X):
        return 1 * (p_val(X) < 0.05)


    # estimate the conditional with logistic regression
    target_p1 = data[M].mean()
    formula = "{} ~ {}".format(M, '+'.join(mpM))
    glm_model = smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit()


    # defining a weight function
    def weight(X):
        if X.__class__ == np.ndarray:
            df = pd.DataFrame(X, columns=[M] + mpM)
        else:
            df = X
        prob_1 = glm_model.predict(df)
        prob = prob_1 * (df[M]) + (1 - prob_1) * (1 - df[M])
        target_prob = target_p1 * (df[M]) + (1 - target_p1) * (1 - df[M])
        w = target_prob / prob
        return w


    # testing the (conditional) independence in the resample
    psi = ShiftTester(weight, T, replacement=False, verbose=False, p_val=p_val)

    X_tune = data[[M] + mpM].to_numpy()
    quantile_repeats = 50
    cutoff = np.quantile(np.random.uniform(size=(1000, quantile_repeats)).min(axis=1), 0.05)

    m = psi.tune_m(X_tune, cond=None, j_x=list(range(1, X_tune.shape[1])), j_y=[0], logistic=True, m_init=50,
                   m_factor=1.5,
                   repeats=quantile_repeats, p_cutoff=cutoff)

    resample_df = psi.resample(data, replacement="REPL-reject", m=m)
    # p-value for M ind. (A, C, Z) in the resample
    pval_M_resample = dHSIC.dhsic_test(X=resample_df[M].to_numpy(),
                                       Y=resample_df[mpM].to_numpy()).rx2('p.value')[0]
    # p-value for Z ind. Y | C in the resample
    pval_Y_resample = psi.combination_test(data, replacement="REPL-reject", m=m, n_combinations=10, method="hartung")

    # testing the (conditional) independence in the original sample
    yvec = data[[Y]].to_numpy()
    mvec = data[M].to_numpy()
    mpMvec = data[mpM].to_numpy()
    zvec = data[[Z]].to_numpy()
    cvec = data[C].to_numpy()

    # p-value for M ind. (A, C, Z) in the original sample
    pval_M_origin = dHSIC.dhsic_test(X=mvec, Y=mpMvec, method='gamma').rx2('p.value')[0]
    # p-value for Z ind. Y | C in the original sample
    pval_Y_original = CondIndTests.KCI(yvec, zvec, cvec).rx2('pvalue')
