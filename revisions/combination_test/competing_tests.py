from statsmodels.stats.proportion import proportion_confint
import rpy2.robjects.packages as packages

# Import rpy for testing with R functions
import rpy2.robjects as robj
CondIndTests = packages.importr("CondIndTests")
GeneralisedCovarianceMeasure = packages.importr("GeneralisedCovarianceMeasure")
dHSIC = packages.importr("dHSIC")


def to_r(x): return robj.FloatVector(x)


# Kernel Conditional Independence
def get_KCI(args, return_p_val=False):
    def KCI(X):
        pval = (CondIndTests.KCI(to_r(X[:,args.Z_pos]), to_r(X[:,2]), to_r(X[:,1])).rx2('pvalue'))[0]
        if return_p_val:
            return pval
        else:
            return 1.0*(pval < args.alpha)

# GeneralisedCovarianceMeasure
def get_GCM(args, return_p_val=False):
    def GCM(X):
        pval = (GeneralisedCovarianceMeasure.gcm_test(X=to_r(X[:,args.covariate_pos]), Z=to_r(X[:,args.Z_pos]), Y=to_r(X[:,args.response_pos]), regr_method="gam").rx2('p.value'))[0]
        if return_p_val:
            return pval
        else:
            return 1.0*(pval < args.alpha)
    return GCM