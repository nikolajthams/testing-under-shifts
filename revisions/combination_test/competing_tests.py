from statsmodels.stats.proportion import proportion_confint
import rpy2.robjects.packages as packages
from revisions.combination_test.data_generation import scm
from revisions.combination_test.arguments import get_args
from fcit import fcit
from pcit.IndependenceTest import PCIT

# Import rpy for testing with R functions
import rpy2.robjects as robj
CondIndTests = packages.importr("CondIndTests")
GeneralisedCovarianceMeasure = packages.importr("GeneralisedCovarianceMeasure")
dHSIC = packages.importr("dHSIC")
def to_r(x): return robj.FloatVector(x)

def ensure_2d(x):
    if x.ndim == 1:
        return x.reshape(-1, 1)
    else:
        return x

# Kernel Conditional Independence
def get_KCI(args, return_p_val=False):
    def KCI(X):
        pval = (CondIndTests.KCI(to_r(X[:,args.Z_pos]), to_r(X[:,2]), to_r(X[:,1])).rx2('pvalue'))[0]
        if return_p_val:
            return pval
        else:
            return 1.0*(pval < args.alpha)
    return KCI

# GeneralisedCovarianceMeasure
def get_GCM(args, return_p_val=False):
    def GCM(X):
        pval = (GeneralisedCovarianceMeasure.gcm_test(X=to_r(X[:,args.covariate_pos]), Z=to_r(X[:,args.Z_pos]), Y=to_r(X[:,args.response_pos]), regr_method="gam").rx2('p.value'))[0]
        if return_p_val:
            return pval
        else:
            return 1.0*(pval < args.alpha)
    return GCM

def get_fcit(args, return_p_val=False):
    def fcit_test(X):
        pval = fcit.test(
                    ensure_2d(X[:,args.covariate_pos]), 
                    ensure_2d(X[:,args.response_pos]), 
                    ensure_2d(X[:,args.Z_pos])
        )
        if return_p_val:
            return pval
        else:
            return 1.0*(pval < args.alpha)
    return fcit_test

def get_pcit(args, return_p_val=False):
    def pcit_test(X):
        pval = PCIT(
            ensure_2d(X[:,args.covariate_pos]), 
            ensure_2d(X[:,args.response_pos]), 
            z=ensure_2d(X[:,args.Z_pos])
        )[0][0]
            
        if return_p_val:
            return pval
        else:
            return 1.0*(pval < args.alpha)
    return pcit_test


if __name__ == "__main__":
    args = get_args()
    pv_gcm = get_GCM(args, return_p_val=True)
    pv_kci = get_KCI(args, return_p_val=True)
    pv_pcit = get_pcit(args, return_p_val=True)
    pv_fcit = get_fcit(args, return_p_val=True)

    X = scm(2000)
    # Time each algorithm
    import time
    start = time.time()
    pval = pv_gcm(X)
    end = time.time()
    print(f"GCM: {end - start}, p-val {pval}")
    start = time.time()
    pval = pv_kci(X)
    end = time.time()
    print(f"KCI: {end - start}, p-val {pval}")
    start = time.time()
    pval = pv_pcit(X)
    end = time.time()
    print(f"PCIT: {end - start}, p-val {pval}")
    start = time.time()
    pval = pv_fcit(X)
    end = time.time()
    print(f"FCIT: {end - start}, p-val {pval}")



