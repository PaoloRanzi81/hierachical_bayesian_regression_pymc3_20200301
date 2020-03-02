"""
TITLE: "Bayesian Logistic Regression + MCMC  + marketing data-set"
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.7

DESCRIPTION: 
Please change the following sections according to your individidual input preferences:
    - 'SETTING PATHS AND KEYWORDS'; 
    - 'PARAMETERS TO BE SET!!!'

"""

###############################################################################
## 1. IMPORTING LIBRARIES
# import required Python libraries
import platform
import os
import datetime
import numpy as np
import pandas as pd
from copy import deepcopy
import pymc3 as pm
import theano
import joblib
from multiprocessing import cpu_count
from sklearn.preprocessing import LabelEncoder  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az


###############################################################################
## 2. SETTING PATHS AND KEYWORDS
# In order to set the correct pathways/folders, check which system are you
# using. It should be either Linux laptop (release == '5.0.0-29-generic') 
# or Linux server (release == '4.4.0-143-generic').
RELEASE = platform.release()

if RELEASE == '5.3.0-28-generic': # Linux laptop
   BASE_DIR_INPUT = ('/media/paolo/4C0F-08B1/analyses/python/bayesian_logistic/raw_data')
   BASE_DIR_OUTPUT = ('/media/paolo/4C0F-08B1/analyses/python/bayesian_logistic/outputs')

else:
   BASE_DIR_INPUT = ('/home/ubuntu/raw_data')
   BASE_DIR_OUTPUT = ('/home/ubuntu/outputs')

   
###############################################################################
## 3. PARAMETERS TO BE SET!!!
input_file_name_1 = ('dataset_marketing.csv')
output_file_name_6 = None 
#input_file_name_1 = ('dataset_sample_1000.csv') 
#output_file_name_1 = ('dataset_sample_1000.csv') 
output_file_name_2 = ('047_analysis_bayesian_model.joblib') 
output_file_name_3 = ('047_analysis_trace_model.joblib') 
output_file_name_4 = ('047_analysis_X_test_oob.csv') 
output_file_name_5 = ('047_analysis_y_test_oob.csv')
#output_file_name_6 = ('clean_dataset.csv')
#input_file_name_1 = None


# setting PyMC3 parameters (IDEAL)
# IDEAL: 440000 draws + tune
draws = 2000
chains = int(round((cpu_count() - 1), 0)) # IDEAL: many chain as many cores
tune = (draws*10) # ideal: 90 % burn-in (also called "tune")
cores = int(round((cpu_count() - 1), 0))

# start clocking time
start_time = time.time()


###############################################################################
## 4. LOADING DATA-SET 
if output_file_name_6 is not None:
    # loading the .csv file 
    X_tmp = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                     output_file_name_6]), header = 0) 
else: 
    # loading the .csv file 
    X_tmp = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                 input_file_name_1]), header = 0)

    
# randomly sampling half of the data-set
#X = X_tmp.sample(frac = 0.05) 
X = X_tmp.copy()  

# free-up RAM memory
X_tmp = None

    
###############################################################################
## 5. SUMMARY STATISTICS
# summary_stats
summary_stats = X.describe()
    
    
###############################################################################
## 6. PRE-PROCESSING 

# drop rows with NaN
X.dropna(axis = 0, inplace = True)

# drop duplicates
X.drop_duplicates(inplace = True)

# target variable
y = X.pop('target_var').to_numpy(dtype="int")

# drop duplicate columns + strings but keep client_ID
X.drop(columns = ['var_10', 'var_11', 'var_12'], axis = 1, inplace = True)
   
# Label ecoding hierachicals 
le_train = LabelEncoder()
X.loc[:, 'var_01'] = le_train.fit_transform(X.loc[:, 'var_01'])

# deep copy
X_tmp_2 = X.copy()
X = None

# selecting 8 variables only (the most important according to ML feature importance + client_ID)
X = X_tmp_2.loc[: , ['var_01', 'var_02', 'var_03', 'var_04', 'var_05', 'var_06', 
                     'var_07', 'var_08', 'var_09']]

# free-up RAM memory
X_tmp_2 = None


# split data into train and test sets. P.S.: Despite Machine Learning/Deep Learning, 
# within a Bayesian framework cross-validation is not performed. Therefore, the data-set
# will not be divided in test-set, validation-set and hold-out-set. Thus, validation-set
# will be skipped.   
X_train_1, X_test_oob_1, y_train_1, y_test_oob_1 = train_test_split(X, y, 
    test_size = 0.05, random_state = None, shuffle = True, stratify = y)

# deepcopy in order to avoid Pandas' warnings
X_train = deepcopy(X_train_1)
X_test_oob = deepcopy(X_test_oob_1)
y_train = deepcopy(y_train_1)
y_test_oob = deepcopy(y_test_oob_1)

# free-up RAM memory
X_train_1 = []
X_test_oob_1 = []
y_train_1 = []
y_test_oob_1 = []

# save not standardized train hold-out-sets as .csv files
X_test_oob.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                     output_file_name_4]), index= False)
np.savetxt(os.path.sep.join([BASE_DIR_OUTPUT, 
                                     output_file_name_5]), y_test_oob, delimiter=',')

# hierachical types 
hierachical_type = X_train.loc[:, 'var_01'].unique() 

# build z_shared_variable
hierachical_variable_train = X_train.pop('var_01')

# standardize necessary step, otherwise PyMC3 throws errors)
ss_train = StandardScaler()
columns_to_be_standardized = ['var_02', 'var_03', 'var_04', 'var_05', 
                              'var_06', 'var_07', 'var_08']
X_train.loc[:, columns_to_be_standardized] = ss_train.fit_transform(X_train.loc[:, columns_to_be_standardized]) 


###############################################################################
## 8. BAYESIAN + MCMC LINEAR REGRESSION  

# set shared theano variables
y_data = deepcopy(y_train)
x_1_data = hierachical_variable_train.to_numpy(dtype="int")
x_2_data = X_train.loc[:, 'var_02'].to_numpy(dtype="int")
x_3_data = X_train.loc[:, 'var_03'].to_numpy(dtype="int")
x_4_data = X_train.loc[:, 'var_04'].to_numpy(dtype="int")
x_5_data = X_train.loc[:, 'var_05'].to_numpy(dtype="int")
x_6_data = X_train.loc[:, 'var_06'].to_numpy(dtype="int")
x_7_data = X_train.loc[:, 'var_07'].to_numpy(dtype="int")
x_8_data = X_train.loc[:, 'var_08'].to_numpy(dtype="int")

# build model 
with pm.Model() as glm_model:   
   
    # create data
    y_shared = pm.Data('y_shared', y_data)
    x_1_shared = pm.Data('x_1_shared', x_1_data)
    x_2_shared = pm.Data('x_2_shared', x_2_data) 
    x_3_shared = pm.Data('x_3_shared', x_3_data) 
    x_4_shared = pm.Data('x_4_shared', x_4_data) 
    x_5_shared = pm.Data('x_5_shared', x_5_data) 
    x_6_shared = pm.Data('x_6_shared', x_6_data) 
    x_7_shared = pm.Data('x_7_shared', x_7_data) 
    x_8_shared = pm.Data('x_8_shared', x_8_data) 
        
    # COMPLEX MODEL: 
    # Priors
    mu_a = pm.Normal('mu_a', mu = 0., sd = 2)
    sigma_a = pm.HalfCauchy('sigma_a', 5)
    mu_b_1 = pm.Normal('mu_b_1', mu = 0., sd = 2)
    sigma_b_1 = pm.HalfCauchy('sigma_b_1', 5)
    mu_b_2 = pm.Normal('mu_b_2', mu = 0., sd = 2)
    sigma_b_2 = pm.HalfCauchy('sigma_b_2', 5)
    mu_b_3 = pm.Normal('mu_b_3', mu = 0., sd = 2)
    sigma_b_3 = pm.HalfCauchy('sigma_b_3', 5)
    mu_b_4 = pm.Normal('mu_b_4', mu = 0., sd = 2)
    sigma_b_4 = pm.HalfCauchy('sigma_b_4', 5)
    mu_b_5 = pm.Normal('mu_b_5', mu = 0., sd = 2)
    sigma_b_5 = pm.HalfCauchy('sigma_b_5', 5)
    mu_b_6 = pm.Normal('mu_b_6', mu = 0., sd = 2)
    sigma_b_6 = pm.HalfCauchy('sigma_b_6', 5)
    mu_b_7 = pm.Normal('mu_b_7', mu = 0., sd = 2)
    sigma_b_7 = pm.HalfCauchy('sigma_b_7', 5)
    mu_b_8 = pm.Normal('mu_b_8', mu = 0., sd = 2)
    sigma_b_8 = pm.HalfCauchy('sigma_b_8', 5)
    
    # Random intercepts
    a = pm.Normal('a', mu = mu_a, sd = sigma_a, shape = 1)
    
    # Random slopes
    b_1 = pm.Normal('b_1', mu = mu_b_1, sd = sigma_b_1, shape = 1) 
    b_2 = pm.Normal('b_2', mu = mu_b_2, sd = sigma_b_2, shape = 1)
    b_3 = pm.Normal('b_3', mu = mu_b_3, sd = sigma_b_3, shape = 1) 
    b_4 = pm.Normal('b_4', mu = mu_b_4, sd = sigma_b_4, shape = 1) 
    b_5 = pm.Normal('b_5', mu = mu_b_5, sd = sigma_b_5, shape = 1) 
    b_6 = pm.Normal('b_6', mu = mu_b_6, sd = sigma_b_6, shape = 1) 
    b_7 = pm.Normal('b_7', mu = mu_b_7, sd = sigma_b_7, shape = 1) 
    b_8 = pm.Normal('b_8', mu = mu_b_8, sd = sigma_b_8, shape = 1) 
    
    # Model error
    sigma_y = pm.HalfCauchy('sigma_y', 5)
       
    # Expected value
    y_hat = (a + b_1*x_1_shared + b_2*x_2_shared + b_3*x_3_shared + b_4*x_4_shared +  
               b_5*x_5_shared + b_6*x_6_shared + b_7*x_7_shared + b_8*x_8_shared)

    # Data likelihood
    #y_like = pm.Normal('y_like', mu = y_hat, sd = sigma_y, observed = y_shared) # GLM (linear regression)
    #y_like = pm.StudentT('y_like', nu = 2.5, mu = y_hat, sigma = sigma_y, observed = y_shared) # robust linear regression 
    y_like = pm.Bernoulli('y_like', logit_p = y_hat, observed = y_shared) # logistic regression!

    # run MCMC
    glm_model_trace = pm.sample(draws = draws, chains = chains, tune = tune, cores = cores)


###############################################################################
## 9. PICKLING THE MODEL BY JOBLIB 

# dump trace model     
joblib.dump(glm_model, os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_2])) 

# dump trace model     
joblib.dump(glm_model_trace, os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_3])) 
 

###############################################################################
## 10. MCMC TRACE DIAGNOSTICS [to be done only once for calibrating the bayesian model]  

## show traces
pm.traceplot(glm_model_trace)  

#az.plot_trace(glm_model_trace, compact=True)
az.plot_trace(glm_model_trace, var_names = "a__0", divergences = "bottom")

# save figure 
date = str(datetime.datetime.now()) 

# to get the current figure...       
fig = plt.gcf()

# save figure
fig.savefig(os.path.sep.join([BASE_DIR_OUTPUT, date[0:10] 
+ "_" + date[11:12] + "_" + date[14:15] + date[17:22] + ".svg"])) 

# close pic in order to avoid overwriting with previous pics
fig.clf()    

## show posterior    
pm.plot_posterior(glm_model_trace)  
# save figure 
date = str(datetime.datetime.now()) 

# to get the current figure...       
fig = plt.gcf()

# save figure
fig.savefig(os.path.sep.join([BASE_DIR_OUTPUT, date[0:10] 
+ "_" + date[11:12] + "_" + date[14:15] + date[17:22] + ".svg"])) 

# close pic in order to avoid overwriting with previous pics
fig.clf() 


## summary statistics
summary_statistics_trace = pm.summary(glm_model_trace)


# shows execution time
print( time.time() - start_time, "seconds")
    
    
