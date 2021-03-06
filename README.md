# Statistical Testing under Distributional Shifts

This reposotory provide resampling methods described in [Statistical Testing under Distributional Shifts](https://arxiv.org/abs/2105.10821), along the code needed to reproduce the experiments in the paper. 

### Installation
To install, run `conda develop .`, to install locally as a package. 

### Use
To use, 
1. Define a test statistic returning `0` (reject hypothesis) or `1` (accept hypthesis). 
    ```python
    import statsmodels.api as sm
    def T(X): 
        return 1*(sm.OLS(X[:, 2], sm.tools.add_constant(X[:, 0])).fit().pvalues[1] < 0.05)
    ```
2. Define a weight function
    ```python
    from scipy.stats import norm
    p = norm.pdf
    def weight(X, scale=1): 
        return p(X[:, 1], scale=scale, loc=0)/p(X[:, 1], loc=X[:,0], scale=sigma_eps)
    ```

3. Once test and weight are defined, instantiate the `ShiftTester` and test hypothesis in the shifted distribution
    ```python
    from resample_and_test import ShiftTester
    psi = ShiftTester(weight, T, replacement=False, verbose=False)
    psi.test(X, replacement="REPL-reject", m=20)
    ```
