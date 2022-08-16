import numpy as np

def e(n=1, mu=0): 
    return np.random.normal(size=(n, 1), loc=mu)  # Gaussian noise
def coin(n, p=0.5):
    return 1*(np.random.uniform(size=(n,1)) < 0.5)
def cb(*args): 
    return np.concatenate(args, axis=1)  # Col bind

# Simulate data from a linear SCM
def linear_scm(n, causal_effect=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    C = e(n) # Bimodal Gaussian distribution
    X = C + 2*e(n)
    Y = C + causal_effect*X + 2*e(n)
    return cb(C, X, Y)

# Simulate data from the scm in the paper
def paper_scm(n, causal_effect=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = e(n, mu=(4*coin(n)-2)) # Bimodal Gaussian distribution
    C = -X**2 + 2*e(n)
    Y = np.sin(C) + causal_effect*X**2 + 2*e(n)
    return cb(X, C, Y)

# Work in progress non-linear SCM
def scm(n, causal_effect=0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = 2*e(n)
    C = np.abs(X-1) + 4*e(n)
    Y = C + causal_effect*X + 2*e(n)
    return cb(X, C, Y)

