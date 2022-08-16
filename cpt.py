import numpy as np
from statsmodels.gam.api import GLMGam, BSplines

class CPT:
    def __init__(self, model, T=None, n_step = 50, M = 1000):
        self.model = model
        self.n_step = n_step
        self.M = M
        
        if T is not None:
            self.T = T
        else:
            def T_(x, y, z=None):
                return np.abs(np.corrcoef(x, y)[0, 1])
            self.T = T_

    # generate CPT copies of X when the conditional distribution is Gaussian
    # i.e. X | Z=Z_i ~ N(mu[i],sig2[i])
    def generate_X_CPT_gaussian(self, X0, mu, sig2):
        log_lik_mat = - np.power(X0,2)[:,None] * (1/2/sig2)[None,:] + X0[:,None] * (mu/sig2)[None,:]
        Pi_mat = self.generate_X_CPT(log_lik_mat)
        return X0[Pi_mat]

    # generate CPT copies of X in general case
    # log_lik_mat[i,j] = q(X[i]|Z[j]) where q(x|z) is the conditional density for X|Z
    def generate_X_CPT(self, log_lik_mat,Pi_init=[]):
        n = log_lik_mat.shape[0]
        if len(Pi_init)==0:
            Pi_init = np.arange(n,dtype=int)
        Pi_ = self.generate_X_CPT_MC(log_lik_mat,Pi_init)
        Pi_mat = np.zeros((self.M,n),dtype=int)
        for m in range(self.M):
            Pi_mat[m] = self.generate_X_CPT_MC(log_lik_mat,Pi_)
        return Pi_mat

    def generate_X_CPT_MC(self,log_lik_mat,Pi):
        n = len(Pi)
        npair = np.floor(n/2).astype(int)
        for _ in range(self.n_step):
            perm = np.random.choice(n,n,replace=False)
            inds_i = perm[0:npair]
            inds_j = perm[npair:(2*npair)]
            # for each k=1,...,npair, decide whether to swap Pi[inds_i[k]] with Pi[inds_j[k]]
            log_odds = log_lik_mat[Pi[inds_i],inds_j] + log_lik_mat[Pi[inds_j],inds_i] \
                - log_lik_mat[Pi[inds_i],inds_i] - log_lik_mat[Pi[inds_j],inds_j]
            swaps = np.random.binomial(1,1/(1+np.exp(-np.maximum(-500,log_odds))))
            Pi[inds_i], Pi[inds_j] = Pi[inds_i] + swaps*(Pi[inds_j]-Pi[inds_i]), Pi[inds_j] - \
                swaps*(Pi[inds_j]-Pi[inds_i])   
        return Pi
    
    def get_p_val(self, X, Y, Z):
        # Fit model of X|Z
        mu = self.model.fit_and_predict(X, Z)
        sig2 = np.repeat(np.var(X - mu), X.shape)

        # Draw permutation using Gaussian assumption
        X_CPT = self.generate_X_CPT_gaussian(X, mu, sig2)
        
        # Compute test statistic in every permutation
        T_list = np.array([self.T(X_CPT[m], Y) for m in range(self.M)])

        # Compare test statistic to observed distribution
        p_val = (1 + (self.T(X, Y) < T_list).sum())/(1 + self.M)
        return p_val


class Model_GAM:
    def fit_and_predict(self, X, Z):
        self.model = GLMGam(X, smoother=BSplines(Z, df = 10, degree=3, include_intercept=True)).fit()
        return self.model.fittedvalues
