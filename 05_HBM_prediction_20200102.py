"""
TITLE: "Model validation by prediction of Bayesian Logistic model"
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
import datetime
import numpy as np
import pandas as pd
from copy import deepcopy
import pymc3 as pm
import joblib
from multiprocessing import cpu_count
from sklearn.preprocessing import LabelEncoder  
#from sklearn.preprocessing import PowerTransformer
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, roc_curve
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

if RELEASE == '5.3.0-40-generic': # Linux laptop
   BASE_DIR_INPUT = ('/media/paolo/4C0F-08B1/analyses/python/bayesian_hbm/raw_data')
   BASE_DIR_OUTPUT = ('/media/paolo/4C0F-08B1/analyses/python/bayesian_hbm/outputs')

else:
   BASE_DIR_INPUT = ('/home/ubuntu/raw_data')
   BASE_DIR_OUTPUT = ('/home/ubuntu/outputs')

   
###############################################################################
## 3. PARAMETERS TO BE SET!!!

# setting PyMC3 parameters 
samples = 10000 # for the predictive part of the script. P.S.: samples < len(y_test_oob)

# load data-set
input_file_name_1 = ('dataset_marketing.csv')

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
    # 4. LOAD PREVIOUSLY TRAINED MODEL AND HOLD-OUT DATA-SET
    ## load hold-out-set
    ## loading the .csv file 
    X_test_oob_1 = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                             output_file_name_4]), header = 0)
        
    y_test_oob = pd.Series(np.loadtxt(os.path.sep.join([BASE_DIR_OUTPUT, 
                                             output_file_name_5]), delimiter=','))
        
    # load objects 
    varying_intercept_slope_noncentered = joblib.load(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_2])) 
    
    
    varying_intercept_slope_noncentered_trace = pm.load_trace(directory = os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_7]), 
                                                              model = varying_intercept_slope_noncentered)
    
    ## load pickled model
    #with open(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_8]), 'rb') as buff:
    #    data = pickle.load(buff)  
    
    #basic_model, varying_intercept_slope_noncentered_trace = data['model'], data['trace']  
    
    # rename
    y_test_oob.rename('truth', inplace = True)
       
    # to be used when 'mod99_cap_member_id' is not present
    X_test_oob = X_test_oob_1.copy()   
    
    # to be used when 'mod99_cap_member_id' is present in the hold-out-set. It merges client_ID to X_test_obb
    X_tmp_1 = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                             input_file_name_1]), header = 0) 
    X_tmp = X_tmp_1.loc[:, ['target_var', 'var_09']] 
    X_tmp_1 = []
    X_test_oob = pd.merge(X_test_oob_1, X_tmp, how ='inner', 
                         on = 'var_09')
    
    # coupon types 
    coupon_type = X_test_oob.loc[:, 'var_01'].unique() 
    
    # Label ecoding coupons 
    le_train = LabelEncoder()
    X_test_oob.loc[:, 'var_01'] = le_train.fit_transform(X_test_oob.loc[:, 'var_01'])
    
    # build z_shared_variable
    hierachical_variable_test_oob = X_test_oob.loc[:,'var_01']
    
    # standardize hold-out-test set
    ss_test_oob = StandardScaler()
    columns_to_be_standardized = ['var_02', 'var_03', 'var_04', 'var_05', 
                                  'var_06', 'var_07', 'var_08']
    X_test_oob.loc[:, columns_to_be_standardized] = ss_test_oob.fit_transform(X_test_oob.loc[:, columns_to_be_standardized]) 
    
    
    ###############################################################################
    ## LOAD OUT-OF-BAG SET
    
    # Build hold-out-set (Pandas Series)
    y_data_1 = pd.Series(np.zeros(y_test_oob.shape[0])).rename('y_data')
    y_data = y_data_1.to_numpy(dtype="int") # try also .T; 
    x_1_data = hierachical_variable_test_oob.to_numpy(dtype="int")
    x_2_data = X_test_oob.loc[:, 'var_02'].to_numpy()
    x_3_data = X_test_oob.loc[:, 'var_03'].to_numpy()
    x_4_data = X_test_oob.loc[:, 'var_04'].to_numpy()
    x_5_data = X_test_oob.loc[:, 'var_05'].to_numpy()
    x_6_data = X_test_oob.loc[:, 'var_06'].to_numpy()
    x_7_data = X_test_oob.loc[:, 'var_07'].to_numpy()
    x_8_data = X_test_oob.loc[:, 'var_08'].to_numpy()
    
    
    ###############################################################################
    ## SETTING MODEL (e.g. VARYING_INTERCEPT_AND_SLOPE (NON-CENTERED))
    
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
             
        return varying_intercept_slope_noncentered
    
    
    ###############################################################################
    ## PREDICTING 
    
    # predicting new values from the posterior distribution of the previously trained model
    with model_factory(x_2_data, x_3_data, x_4_data, x_5_data, x_6_data, x_7_data, 
                       x_8_data, y_data, x_1_data) as test_model:
       trace_df_1 = pm.trace_to_dataframe(varying_intercept_slope_noncentered_trace, include_transformed=True)
       
       # all variables
       post_pred_big = pm.sample_posterior_predictive(trace = varying_intercept_slope_noncentered_trace,
                                         samples = samples) 
       print('post_pred_big shape', post_pred_big['y_like'].shape)
       

    ###############################################################################
    ## PREDICTION PERFORMANCE
    # check number of predicted '1'    
    booked_sum = y_test_oob.sum().astype(int)
    print('number_of_bookings = %.i' % (booked_sum))
    
    # sort output values from PyMC3 prediction 
    transposed_output = pd.DataFrame(post_pred_big['y_like'])
    transposed_output_sorted = transposed_output.sum(axis = 0)
    transposed_output_sorted.sort_values(axis = 0, inplace = True, ascending = False)
    transposed_output_sorted.rename('probability', inplace = True)
    transposed_output_sorted_tmp = transposed_output_sorted.reset_index()
    
    # create binary output
    ones_padding = pd.Series(np.ones(booked_sum))
    zeros_padding = pd.Series(np.zeros(len(transposed_output_sorted) - booked_sum))
    binary_output = pd.concat([ones_padding, zeros_padding], axis = 0).rename('binary', inplace = True)
    binary_output.reset_index(drop = True, inplace = True)
    
    # concatenate
    #X_tmp = X_tmp_1.loc[:, ['mod00_booking_yn', 'mod99_cap_member_id']] 
    output = pd.concat([transposed_output_sorted_tmp, binary_output], axis = 1)
    output.set_index('index', inplace = True)
    y_predicted = pd.concat([output, y_test_oob, X_test_oob.loc[:, 'mod99_cap_member_id']], axis = 1)
    
    # compute and print classification metrics                    
    MCC_metric = matthews_corrcoef(y_predicted.loc[:, 'truth'], y_predicted.loc[:, 'binary'])
    precision_metric = precision_score(y_predicted.loc[:, 'truth'], y_predicted.loc[:, 'binary'])
    recall_metric = recall_score(y_predicted.loc[:, 'truth'], y_predicted.loc[:, 'binary'])
    print('MCC = %.3f\n precision = %.3f\n recall = %.3f' % \
              (MCC_metric, precision_metric, recall_metric ))  

"""
###############################################################################
## 10. MCMC TRACE DIAGNOSTICS [to be done only once for calibrating the bayesian model]  


# see graph for model
import graphviz
pm.model_to_graphviz(varying_intercept_slope_noncentered)

# too RAM damanding
data = az.convert_to_dataset(varying_intercept_slope_noncentered_trace)

## show traces
pm.traceplot(varying_intercept_slope_noncentered_trace)  

#az.plot_trace(glm_model_trace, compact=True)
az.plot_trace(varying_intercept_slope_noncentered_trace[:3000], var_names = "υ", divergences = "bottom")
az.plot_trace(varying_intercept_slope_noncentered_trace, var_names = "sigma_a", divergences = "bottom")

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
pm.plot_posterior(varying_intercept_slope_noncentered_trace)  
# save figure 
date = str(datetime.datetime.now()) 

# to get the current figure...       
fig = plt.gcf()

# save figure
fig.savefig(os.path.sep.join([BASE_DIR_OUTPUT, date[0:10] 
+ "_" + date[11:12] + "_" + date[14:15] + date[17:22] + ".svg"])) 

# close pic in order to avoid overwriting with previous pics
fig.clf() 

# forest plot
forestplot(varying_intercept_slope_noncentered_trace, varnames=['b']);


## summary statistics
summary_statistics_trace_083 = pm.summary(varying_intercept_slope_noncentered_trace)


""" 


# shows execution time
print( time.time() - start_time, "seconds")




