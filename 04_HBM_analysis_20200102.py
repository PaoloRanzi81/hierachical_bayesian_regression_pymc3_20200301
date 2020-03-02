"""
TITLE: "Bayesian Hierachical Bayesian Modelling (HBM) + MCMC + marketing data-set"
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.7

DESCRIPTION: 
Further, please change the following sections according to your individidual input preferences:
    - 'SETTING PATHS AND KEYWORDS'; 
    - 'PARAMETERS TO BE SET!!!'

"""
###############################################################################
## 1. IMPORTING LIBRARIES
# import required Python libraries
import platform
import os
import numpy as np
import pandas as pd
from copy import deepcopy
import pymc3 as pm
import joblib
from multiprocessing import cpu_count
from sklearn.preprocessing import LabelEncoder  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time


###############################################################################
## 2. SETTING PATHS AND KEYWORDS
# In order to set the correct pathways/folders, check which system are you
# using. It should be either Linux laptop (release == '5.0.0-29-generic') 
# or Linux server (release == '4.4.0-143-generic').
RELEASE = platform.release()

if RELEASE == '5.3.0-40-generic': # Linux laptop
   BASE_DIR_INPUT = ('/media/paolo/4C0F-08B1/analyses/python/bayesian_hbm/raw_data')
   BASE_DIR_OUTPUT = ('/media/paolo/4C0F-08B1/analyses/python/bayesian_hbm/outputs')

else:
   BASE_DIR_INPUT = ('/home/ubuntu/raw_data')
   BASE_DIR_OUTPUT = ('/home/ubuntu/outputs')  


###############################################################################
## 3. PARAMETERS TO BE SET!!!

# number of chunks (splits and chunks have the same meaning here)
number_of_splits = 5    

# setting PyMC3 parameters (IDEAL)
# ideal: 440000 draws
draws = 2000
chains = int(round((cpu_count() - 1), 0)) # IDEAL: many chain as many cores
tune = (draws*10)
#tune = (draws*90)/100 # ideal: 90 % burn-in (also called "tune")
cores = int(round((cpu_count() - 1), 0))

# start clocking time
start_time = time.time()

    
###############################################################################
## FOR LOOP: COMPUTING HBM WITH EACH CHUNK
# build a list of chunks' names
for chunk in range(0, number_of_splits): 
            
    # generate chunk's name
    chunk_name = ("00%s_chunk.csv" % (chunk+1))
    
    # build outputs' names
    output_file_name_2 = ('120_%s_analysis_bayesian_model.joblib' % (chunk_name[0:9])) 
    output_file_name_4 = ('120_%s_analysis_X_test_oob.csv' % (chunk_name[0:9])) 
    output_file_name_5 = ('120_%s_analysis_y_test_oob.csv' % (chunk_name[0:9]))
    output_file_name_7 = ('120_%s_analysis_saved_trace' % (chunk_name[0:9]))
       
    
    ###############################################################################
    ## 4. LOADING DATA-SET 
    X_tmp = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                     chunk_name]), header = 0) 
    
    # deep copy
    X = X_tmp.copy()  
    
    # free-up RAM memory
    X_tmp = None
        
        
    ###############################################################################
    ## 6. PRE-PROCESSING 
    # TEST: double-check columns' names
    #X_columns_1 = pd.DataFrame(X.columns)
    
    # drop rows with NaN
    X.dropna(axis = 0, inplace = True)
    
    # drop duplicates
    X.drop_duplicates(inplace = True)
    
    # target variable 'mod64_ratio_of_bookings_with_coupon'
    y = X.pop('target_var').to_numpy(dtype="int")
    
    # drop duplicate columns + strings but keep client_ID
    X.drop(columns = ['var_10', 'var_11', 'var_12'], axis = 1, inplace = True)
       
    # Label ecoding coupons 
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
    
    # coupon types 
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
    x_1_data = hierachical_variable_train.to_numpy(dtype="int") # it works both with or without .T
    x_2_data = X_train.loc[:, 'var_02'].to_numpy()
    x_3_data = X_train.loc[:, 'var_03'].to_numpy()
    x_4_data = X_train.loc[:, 'var_04'].to_numpy()
    x_5_data = X_train.loc[:, 'var_05'].to_numpy()
    x_6_data = X_train.loc[:, 'var_06'].to_numpy()
    x_7_data = X_train.loc[:, 'var_07'].to_numpy()
    x_8_data = X_train.loc[:, 'var_08'].to_numpy()
    
    
    ###############################################################################
    ## VARYING_INTERCEPT_AND_SLOPE (NON-CENTERED)
    
    def model_factory(x_2_data, x_3_data, x_4_data, x_5_data, x_6_data, x_7_data, 
                       x_8_data, y_data, x_1_data):
        
        with pm.Model() as varying_intercept_slope_noncentered:
          
            # Priors
            mu_a = pm.Normal('mu_a', mu = 0.05, sd = 2)
            sigma_a = pm.HalfCauchy('sigma_a', 5)
            
            mu_b_1 = pm.InverseGamma('mu_b_1', mu = 0.05, sigma = 2)
            sigma_b_1 = pm.HalfCauchy('sigma_b_1', 5)
            mu_b_2 = pm.InverseGamma('mu_b_2', mu = 0.05, sigma = 2)
            sigma_b_2 = pm.HalfCauchy('sigma_b_2', 5)
            mu_b_3 = pm.InverseGamma('mu_b_3', mu = 0.05, sigma = 2)
            sigma_b_3 = pm.HalfCauchy('sigma_b_3', 5)
            mu_b_4 = pm.InverseGamma('mu_b_4', mu = 0.05, sigma = 2)
            sigma_b_4 = pm.HalfCauchy('sigma_b_4', 5)
            mu_b_5 = pm.InverseGamma('mu_b_5', mu = 0.05, sigma = 2)
            sigma_b_5 = pm.HalfCauchy('sigma_b_5', 5)
            mu_b_6 = pm.InverseGamma('mu_b_6', mu = 0.05, sigma = 2)
            sigma_b_6 = pm.HalfCauchy('sigma_b_6', 5)
            mu_b_7 = pm.InverseGamma('mu_b_7', mu = 0.05, sigma = 2)
            sigma_b_7 = pm.HalfCauchy('sigma_b_7', 5)
                   
            # Non-center random intercepts + slopes
            u = pm.Normal('u', mu = 0, sd = 2, shape = len(hierachical_type))
            a = mu_a + u * sigma_a
                   
            # Random slopes
            b_1 = mu_b_1 + u * sigma_b_1
            b_2 = mu_b_2 + u * sigma_b_2
            b_3 = mu_b_3 + u * sigma_b_3
            b_4 = mu_b_4 + u * sigma_b_4
            b_5 = mu_b_5 + u * sigma_b_5
            b_6 = mu_b_6 + u * sigma_b_6
            b_7 = mu_b_7 + u * sigma_b_7
                   
            # Expected value
            y_hat = (a[x_1_data] + b_1[x_1_data]*x_2_data + b_2[x_1_data]*x_3_data + 
                    b_3[x_1_data]*x_4_data + b_4[x_1_data]*x_5_data + b_5[x_1_data]*x_6_data + 
                    b_6[x_1_data]*x_7_data + b_7[x_1_data]*x_8_data) 
                   
            # Data likelihood (discrete distributions only)
            pm.Bernoulli('y_like', logit_p = y_hat, observed = y_data)
            
        # dump trace model     
        joblib.dump(varying_intercept_slope_noncentered, os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_2]))   
                
        return varying_intercept_slope_noncentered
    
    
    with model_factory(x_2_data, x_3_data, x_4_data, x_5_data, x_6_data, x_7_data, 
                       x_8_data, y_data, x_1_data) as train_model:
        
        # run MCMC
        varying_intercept_slope_noncentered_trace = pm.sample(draws = draws, tune = tune, chains = chains, 
                                    cores = cores, target_accept = 0.95, discard_tuned_samples = True) # very slow, but it works.
    
    
    ###############################################################################
    ## 9. PICKLING THE TRACE BY JOBLIB 

    # save the trace (alternative method 1)
    pm.save_trace(trace = varying_intercept_slope_noncentered_trace, 
                  directory = os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_7]),
                                               overwrite = True)

    
# shows execution time
print( time.time() - start_time, "seconds")




      
    
    
    
    
