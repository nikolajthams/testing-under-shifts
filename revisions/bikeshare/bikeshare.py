import pandas as pd
import numpy as np
from urllib.request import urlopen   
import statsmodels.api as sm
import os
from scipy.stats import norm
from resample_and_test import ShiftTester

def download_data(path="zipfile"):
    """
    Download and unzip the data.
    """
    URL = 'https://s3.amazonaws.com/capitalbikeshare-data/2011-capitalbikeshare-tripdata.zip'
    
    url = urlopen(URL)
    output = open(f'{path}.zip','wb')
    output.write(url.read())
    output.close()
    
    df = pd.read_csv(f'{path}.zip')
    os.remove(f'{path}.zip')
    return df

def duration_mean_by_time(group):
        # estimate E[log(Duration)] & Var[log(Duration)] 
    # for each time of day (in minutes), for that particular route
    n = len(group)
    h = 20 # bandwidth in minutes
    K = np.exp(-np.power(12*60 - np.abs(np.abs(np.array(group.Minute)[:,None] - np.array(group.Minute)[None,:])\
               - 12*60),2)/2/h**2) * np.array(group.Month != 10)
    # K(i,j) is nonzero only for j in training set
    group['K_weights_total'] = np.dot(K, np.ones(n))
    group['Duration_mean'] = np.divide(np.dot(K, group.Duration), group.K_weights_total, \
            out=np.zeros(n), where= (group.K_weights_total!=0))
    group['Duration_var'] = np.divide(np.dot(K, np.power(group.Duration,2)), group.K_weights_total, \
            out=np.zeros(n), where= (group.K_weights_total!=0)) - np.power(group.Duration_mean,2)
    return group

def prepare_data():
    df = download_data()
    tmp = df['Start date'].str.split('-',expand=True)
    df['Month'] = tmp[1].astype(int)
    tmp = tmp[2].str.split(' ',expand=True)
    df['Day'] = tmp[0].astype(int)
    tmp = tmp[1].str.split(':',expand=True)
    df['Minute'] = 60 * tmp[0].astype(int) + tmp[1].astype(int)
    df['Day of week'] = (4 + df['Day'] +\
    np.array([0,31,28,31,30,31,30,31,31,30,31,30]).cumsum()[df.Month-1])%7
    df.Duration = np.log(df.Duration)

    # public holidays: 
    # labor day, columbus day, halloween, veterans day, thanksgiving +/- 1 day
    df['Holiday'] = ((df['Month']==9)&(df['Day']==5))| ((df['Month']==10)&(df['Day']==10)) | \
        ((df['Month']==10)&(df['Day']==31))| ((df['Month']==11)&(np.abs(df['Day']-24)<=1))

    df = df[(df['Day of week'] <= 4) & (df.Month >= 9) & (df.Month <= 11)  & (df.Holiday == False)]
    # keep only regular weekdays in sept & nov (train) or oct (test)

    df = df.groupby(['Start station number', 'End station number']).apply(duration_mean_by_time)
    return df[(df.Month == 10) & (df.K_weights_total >= 20)] # test set


if False:
    df1 = prepare_data()
    df1.to_csv('revisions/bikeshare/df1.csv')
else: 
    df1 = pd.read_csv('revisions/bikeshare/df1.csv')

p = norm.pdf
data = pd.DataFrame({"X": df1.Duration, 
                     "Duration_mean": np.array(df1.Duration_mean), 
                     "Duration_var": np.maximum(1e-12,np.array(df1.Duration_var)), 
                     "Y1": np.array(df1['Member type'] == 'Member'), 
                     "Y2": np.array(df1['Day']),
                     "Y3": np.array(df1['Day of week'])})

# Compute weights
def weight(data):
    X, mu, sigma_squared = data['X'], data['Duration_mean'], data['Duration_var']
    
    # Set target mean and variance as 
    mu_target, sigma_squared_target = X.mean(), X.var()

    return p(X,mu_target,np.sqrt(sigma_squared_target)) / p(X,mu,np.sqrt(sigma_squared))


# Wrapper function for getting p-values of hypothesis X \indep Y_name | Z
def get_p_val(Y_name, use_permutation=False):
    # Return this function
    def p_val(data):
        X = data['X']
        Y = data[Y_name]
        
        # Either use a permutation test (same as CPT) or a correlation test
        if use_permutation:
            test_func = lambda x, y: np.abs(np.corrcoef(x, y))[0,1]
            M = 1000
            return (1+np.sum([test_func(X, Y) < test_func(np.random.permutation(X), Y) for _ in range(M)]))/(1+M)
        else:
            return sm.OLS(Y, sm.add_constant(X)).fit().pvalues['X']
    return p_val


np.random.seed(1)
for Y_name in ['Y1','Y2','Y3']:
    psi = ShiftTester(weight, p_val=get_p_val(Y_name), replacement=False)

    n = len(data)
    m = int(2*np.sqrt(n))
    # Use our method to test hypothesis of conditional independence
    pv = psi.combination_test(data, m=m, n_combinations=100, method="hartung")
    print(f"CondIndep: {Y_name}: {pv}")

    # Also print p-values for test of marginal independence
    marg_pval = sm.OLS(data[Y_name], sm.add_constant(data[['Duration_var', 'Duration_mean']])).fit().f_test(["Duration_var", "Duration_mean"]).pvalue
    print(f"MargIndep: {Y_name}: {marg_pval}\n")