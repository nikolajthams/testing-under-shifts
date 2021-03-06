import numpy as np
from scipy.stats import norm
import pandas as pd
from tqdm import tqdm
import torch
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load kcgof for testing conditional
import kcgof.util as util
import kcgof.cdensity as cden
import kcgof.cgoftest as cgof
import kcgof.kernel as ker


# Help functions
e = lambda n=1: np.random.normal(size=(n,1)) # Gaussian noise
def cb(*args): return np.concatenate(args, axis=1) #Col bind
p = norm.pdf

class ShiftTester():
    """
    Shifted tester for resampling and applying test statistic to resample.

    Inputs:
        - weight: function, taking as input data, X, and returning an array of weights for each row
        - T: function, taking as input data and returning a p-value
        - rate: function, taking as input n, and returning the rate of resampling m
        - replacement: boolean, indicating whether or not resampling is with replacement
        - degenerate: string [raise, retry, ignore], specifying handling of degenerate resamples
    """
    def __init__(self, weight, T, rate=lambda n: n**0.45, replacement=False,
                 degenerate="raise", reject_retries=100, verbose=False,
                 gibbs_steps=10, alternative_sampler=False):
        self.weight, self.rate, self.T = weight, rate, T
        self.replacement = replacement
        self.degenerate = degenerate
        self.reject_retries = reject_retries
        self.gibbs_steps = gibbs_steps
        self.verbose = verbose
        self.alternative_sampler = alternative_sampler

        # When degenerates are handled with retries, we count the retries, as to avoid infinite recursion
        self.retries = 0

        # Initiate latest_resample variable, which allows for optionally storing last resample
        self.latest_resample = None

    def resample(self, X, replacement=None, m=None, store_last=False):
        """
        Resampling function that returns a weighted resample of X
        """

        # Potentially overwrite default replacement
        replacement = replacement if replacement is not None else self.replacement

        # Compute sample and resample size
        n = X.shape[0]
        m = int(self.rate(n)) if m is None else m

        # Draw weights
        if callable(self.weight):
            w = self.weight(X)
        else:
            w = self.weight
        w /= w.sum()

        # Resample with modified replacement scheme:
        # Sample w replace, but reject if non-distinct
        if replacement == "REPL-reject":
            idx = np.random.choice(n, size=m, p=w, replace=True)
            count = 0
            while count < self.reject_retries and (len(np.unique(idx)) != len(idx)):
                count += 1
                idx = np.random.choice(n, size=m, p=w, replace=True)
            if self.verbose:
                print(f"Rejections: {count}")
            if (len(np.unique(idx)) != len(idx)):
                if self.alternative_sampler == "error":
                    raise ValueError("Unable to draw sample from REPL rejection sampler")
                else:
                    return self.resample(X, replacement=self.alternative_sampler, m=m, store_last=store_last)
        elif replacement == "NO-REPL-gibbs":
            # Initialize space
            space = np.arange(n)
            # Initialize Gibbs sampler in NO-REPL distribution and shuffle to mimick dist
            idx = np.random.choice(space, size=m, p=w, replace=False)
            np.random.shuffle(idx)

            # Loop, sampling from conditional
            for _ in range(self.gibbs_steps):
                for j, i in (tqdm(enumerate(idx)) if self.verbose else enumerate(idx)):
                    retain = np.delete(idx, j)
                    vacant = np.setdiff1d(space, retain)
                    idx[j] = np.random.choice(vacant, 1, p=w[vacant]/w[vacant].sum())

        elif replacement == "NO-REPL-reject":
            # Denominator for rejection sampler is smallest weights
            m_smallest = np.cumsum(w[np.argsort(w)][:(m-1)])

            # Sample from proposal, and compute bound p/Mq
            count = 0
            idx = np.random.choice(n, size=m, p=w, replace=False)
            bound = np.prod(1 - np.cumsum(w[idx])[:-1])/np.prod(1 - m_smallest)

            while ((np.random.uniform() > bound) and count < self.reject_retries):
                count += 1
                idx = np.random.choice(n, size=m, p=w, replace=False)
                bound = np.prod(1 - np.cumsum(w[idx])[:-1])/np.prod(1 - m_smallest)

            if count == self.reject_retries:
                if self.alternative_sampler == "error":
                    raise ValueError("Unable to draw sample from NO-REPL rejection sampler")
                else:
                    return self.resample(X, replacement=self.alternative_sampler, m=m, store_last=store_last)

        # If nothing else, just sample with or without replacement
        else:
            idx = np.random.choice(n, size=m, p=w, replace=replacement)

        if isinstance(X, pd.core.frame.DataFrame):
            out = X.loc[idx]
        elif isinstance(X, np.ndarray):
            out = X[idx]
        else:
            raise TypeError("Incorrect dataformat provided. Please provide either a Pandas dataframe or Python array")

        # Handling the situation with only a single data point drawn
        unique_draws = len(np.unique(idx))
        if replacement and unique_draws == 1:
            if self.degenerate == "raise": raise ValueError("Degenerate resample drawn!")
            elif self.degenerate == "retry":
                self.retries = self.retries + 1
                if self.retries < 10:
                    return self.resample(X, replacement)
                else:
                    self.retries = 0
                    raise ValueError("Degenerate resample drawn!")
            elif self.degenerate == "ignore": return out

        if store_last:
            self.latest_resample = out

        return out

    def test(self, X, replacement=None, m=None, store_last=False):
        # Resample data
        X_m = self.resample(X, replacement, m=m, store_last=store_last)

        # Apply test statistic
        return self.T(X_m)

    def kernel_conditional_validity(self, X, cond, j_x, j_y, return_p=False):
        """
        Test that resampled data has the correct conditional

        X:
            Data (resampled) in numpy format

        cond(x,y):
            torch function taking as input torch arrays x, y and
            outputting the log conditional density log p(y|x)

        j_y, j_x:
            Lists specifying which columns in X are respectively y and x.
            E.g. j_x = [0, 1, 2], j_y = [3]

        return_p:
            If True, returns p-value, else 0-1 indicator of rejection
        """

        # Convert input data to torch
        x, y = torch.from_numpy(X[:,j_x]).float(), torch.from_numpy(X[:,j_y]).float()
        dx, dy = len(j_x), len(j_y)

        # Specify conditional model
        cond_ = cden.from_log_den(dx, dy, cond)

        # Choose kernel bandwidth
        sigx = util.pt_meddistance(x, subsample=1000, seed=2)
        sigy = util.pt_meddistance(y, subsample=1000, seed=3)
        k = ker.PTKGauss(sigma2=sigx**2)
        l = ker.PTKGauss(sigma2=sigy**2)

        # Create kernel object
        kcsdtest = cgof.KCSDTest(cond_, k, l, alpha=0.05, n_bootstrap=500)

        # Compute output
        result = kcsdtest.perform_test(x, y)

        if return_p: return result['pvalue']
        return 1*result['h0_rejected']

    def gaussian_validity(self, X, cond, j_x, j_y, const=None, return_p=False):
        """
        Test that resampled data has the correct conditional

        X:
            Data (resampled) in numpy format

        cond:
            List containing linear conditional mean of y|x.
            E.g. if y = 2*x1 + 3*x2, should be cond=[2, 3]

        j_y, j_x:
            Lists specifying which columns in X are respectively y and x.
            E.g. j_x = [0, 1, 2], j_y = [3]

        const:
            Intercept in target distribution. Default is None

        return_p:
            If True, returns p-value, else 0-1 indicator of rejection
        """
        x, y = X[:,j_x], X[:,j_y]
        if const=="fit":
            tests = [f"(x{i+1} = {b})" for i, b in enumerate(cond)]
            p_val = sm.OLS(y, sm.add_constant(x)).fit().f_test(tests).pvalue
        elif const is not None:
            tests = [f"(const = {const})"] + [f"(x{i+1} = {b})" for i, b in enumerate(cond)]
            p_val = sm.OLS(y, sm.add_constant(x)).fit().f_test(tests).pvalue
        else:
            tests = [f"(x{i+1} = {b})" for i, b in enumerate(cond)]
            p_val = sm.OLS(y, x).fit().f_test(tests).pvalue

        if return_p: return p_val
        return 1*(p_val < 0.05)

    def binary_validity(self, X, cond, j_x, j_y, const=None, return_p=False):
        """
        Test that resampled data has the correct conditional

        X:
            Data (resampled) in numpy format

        cond:
            Dictionary {tuple: float} containing conditional distribution of x|y
            - Dictionary keys are all possible outcomes of x, e.g. x1=1, x2=0 -> (1, 0). Ordering of the tuple should be the same as of j_x
            - Dictionary values specify the probability of y=1 given x.

            E.g. if P(y=1|x1=0, x2=0) = 0.3, P(y=1|x1=0, x2=1) = 0.9, cond should be {(0, 0): 0.3, (0, 1): 0.9}

        j_y, j_x:
            Lists specifying which columns in X are respectively y and x.
            E.g. j_x = [0, 1, 2], j_y = [3]

        return_p:
            If True, returns p-value, else 0-1 indicator of rejection
        """
        # Convert X to data frame
        df = pd.DataFrame(X[:,j_y + j_x], columns = ["x"+str(i) for i in j_y + j_x])

        # Setup formula for statsmodels
        formula = f"x{j_y[0]}~" + ":".join(f"C(x{i})" for i in j_x) + "-1"

        # Convert the conditional distribution to log-odds
        log_odds = {k: np.log(p/(1-p)) for k,p in log_odds.items()}

        # Create list of tests to conduct
        tests = [":".join([f"C(x{idx})[{val}]" for idx, val in zip(j_x, outcome)]) + f"={p}" for outcome,p in log_odds.items()]

        # Conduct F-test and get p-value
        p_val = smf.logit(formula=formula, data=df).fit(disp=0).f_test(tests).pvalue

        if return_p: return p_val
        return 1*(p_val < 0.05)

    def tune_m(self, X, cond, j_x, j_y, gaussian=False, binary=False, m_init = None,
               m_factor=2, p_cutoff=0.1, repeats=100, const=None, replacement=None):
        # Initialize parameters
        n = X.shape[0]
        m = int(np.sqrt(n)/2) if m_init is None else m_init
        replacement = self.replacement if replacement is None else replacement

        res = [1]

        # Loop over increasing m as long as level is below 10%
        while (np.min(res) > p_cutoff) and (m < n):
            # m = int(2*m)
            res = []
            for _ in tqdm(range(repeats)) if self.verbose else range(repeats):

                if gaussian:
                    z = self.gaussian_validity(self.resample(X, m=int(min(m_factor*m, n)), replacement=replacement), cond=cond, j_x=j_x, j_y=j_y, const=const, return_p = True)
                elif binary:
                    z = self.binary_validity(self.resample(X, m=int(min(m_factor*m, n)), replacement=replacement), cond=cond, j_x=j_x, j_y=j_y, return_p = True)
                else:
                    z = self.kernel_conditional_validity(self.resample(X, m=int(min(m_factor*m, n)), replacement=replacement), cond=cond, j_x=j_x, j_y=j_y, return_p = True)
                res.append(z)

            if self.verbose:
                print(f"mean {np.min(res)}, m {min(m_factor*m, n)}")

            if (np.min(res) > p_cutoff): m = int(min(m_factor*m, n))

        return m
