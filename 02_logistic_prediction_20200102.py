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
import theano
import joblib
from multiprocessing import cpu_count
from sklearn.preprocessing import LabelEncoder  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, roc_curve
import time
import matplotlib.pyplot as plt


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
output_file_name_2 = ('047_analysis_bayesian_model.joblib') 
output_file_name_3 = ('047_analysis_trace_model.joblib') 
output_file_name_4 = ('047_analysis_X_test_oob.csv') 
output_file_name_5 = ('047_analysis_y_test_oob.csv')

# setting PyMC3 parameters 
samples = 10000 # for the predictive part of the script. P.S.: samples < len(y_test_oob)

# start clocking time
start_time = time.time()


###############################################################################
# 4. LOAD PREVIOUSLY TRAINED MODEL AND HOLD-OUT DATA-SET    
## load hold-out-set
X_test_oob_1 = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         output_file_name_4]), header = 0)
    
y_test_oob = pd.Series(np.loadtxt(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         output_file_name_5]), delimiter=','))
    
# load objects 
glm_model = joblib.load(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_2])) 
glm_model_trace = joblib.load(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_3])) 

# rename
y_test_oob.rename('truth', inplace = True)
   
# to be used when client ID is not present
X_test_oob = X_test_oob_1.copy()   

# to be used when 'mod99_cap_member_id' is present in the hold-out-set. It merges client_ID to X_test_obb
X_tmp_1 = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         input_file_name_1]), header = 0) 
X_tmp = X_tmp_1.loc[:, ['target_var', 'var_09']] 
X_tmp_1 = []
X_test_oob = pd.merge(X_test_oob_1, X_tmp, how ='inner', 
                     on = 'var_09')

# hierachical types 
hierachical_type = X_test_oob.loc[:, 'var_01'].unique() 

# Label ecoding hierachicals 
le_train = LabelEncoder()
X_test_oob.loc[:, 'var_01'] = le_train.fit_transform(X_test_oob.loc[:, 'var_01'])

# build z_shared_variable
hierachical_variable_test_oob = X_test_oob.loc[:,'var_01']


###############################################################################
## 6. MODEL VALIDATION (NOT-HIERACHICAL): test prediction on hold-out set
# standardize hold-out-test set
ss_test_oob = StandardScaler()
columns_to_be_standardized = ['var_02', 'var_03', 'var_04', 'var_05', 
                              'var_06', 'var_07', 'var_08']
X_test_oob.loc[:, columns_to_be_standardized] = ss_test_oob.fit_transform(X_test_oob.loc[:, columns_to_be_standardized]) 

# Build hold-out-set (Pandas Series)
y_data_1 = pd.Series(np.zeros(X_test_oob.shape[0])).rename('y_data')
#y_data_1 = pd.Series(np.zeros(y_test_oob.shape[0])).rename('y_data')
y_data = y_data_1.to_numpy(dtype="int")
x_1_data = hierachical_variable_test_oob.to_numpy(dtype="int")
x_2_data = X_test_oob.loc[:, 'var_02'].to_numpy(dtype="int")
x_3_data = X_test_oob.loc[:, 'var_03'].to_numpy(dtype="int")
x_4_data = X_test_oob.loc[:, 'var_04'].to_numpy(dtype="int")
x_5_data = X_test_oob.loc[:, 'var_05'].to_numpy(dtype="int")
x_6_data = X_test_oob.loc[:, 'var_06'].to_numpy(dtype="int")
x_7_data = X_test_oob.loc[:, 'var_07'].to_numpy(dtype="int")
x_8_data = X_test_oob.loc[:, 'var_08'].to_numpy(dtype="int")

## TEST: manual prediction: convert traces to DataFrame
trace_df = pm.trace_to_dataframe(glm_model_trace, include_transformed=True)
#trace_df = pm.trace_to_dataframe(glm_model_trace, varnames = ['mu_a', 'mu_b', 'sigma_a', 'sigma_b','sigma_y'],
#                                 include_transformed=True)

#TEST: manual prediction: it works with pm.math.sigmoid() ! Same results as pm.sample_posterior_predictive!!!
y_hat =   (trace_df.loc[:,'a__0'].mean() + trace_df.loc[:,'b_1__0'].mean()*x_1_data + trace_df.loc[:,'b_2__0'].mean()*x_2_data + 
           trace_df.loc[:,'b_3__0'].mean()*x_3_data + trace_df.loc[:,'b_4__0'].mean()*x_4_data +  
               trace_df.loc[:,'b_5__0'].mean()*x_5_data + trace_df.loc[:,'b_6__0'].mean()*x_6_data + trace_df.loc[:,'b_7__0'].mean()*x_7_data + 
               trace_df.loc[:,'b_8__0'].mean()*x_8_data + trace_df.loc[:,'sigma_y'].mean())
y_hat_sigmoid = pd.Series(1 / (1 + np.exp(-y_hat))).rename('y_hat_sigmoid', inplace = True)
odds = y_hat/(1 - y_hat)
y_hat_logit = pd.Series(np.log(np.absolute(odds))).rename('y_hat_logit', inplace = True)
y_hat_inv_logit = pd.Series(np.exp(-y_hat)/(1 + np.exp(y_hat))).rename('y_hat_inv_logit', inplace = True) # very similar to the sigmoid

# predict
with glm_model:
    # WARNING: for GLM all the shared variable have to be built inside the "with context", 
    # during the learning step. Instead, for HBM the shared variable has to be built outside
    # the "with context" both for the learning and for the prediction step. 
    pm.set_data({'x_1_shared': x_1_data})  
    pm.set_data({'x_2_shared': x_2_data}) 
    pm.set_data({'x_3_shared': x_3_data})
    pm.set_data({'x_4_shared': x_4_data})  
    pm.set_data({'x_5_shared': x_5_data}) 
    pm.set_data({'x_6_shared': x_6_data}) 
    pm.set_data({'x_7_shared': x_7_data})
    pm.set_data({'x_8_shared': x_8_data})
    pm.set_data({'y_shared': y_data}) 
    post_pred = pm.sample_posterior_predictive(glm_model_trace, samples = samples)
#    # TEST:
#    post_pred = pm.sample_posterior_predictive(trace=trace_df.to_dict('records'),
#                                         samples=len(trace_df))
    print('post_pred shape', post_pred['y_like'].shape)
    
# check number of predicted '1'    
booked_sum = y_test_oob.sum().astype(int)
print('number_of_bookings = %.i' % (booked_sum))

# sort output values from PyMC3 prediction 
transposed_output = pd.DataFrame(post_pred['y_like'])
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
output = pd.concat([transposed_output_sorted_tmp, binary_output], axis = 1)
output = pd.concat([transposed_output_sorted_tmp, binary_output], axis = 1)
output.set_index('index', inplace = True)
y_predicted = pd.concat([output, y_test_oob, X_test_oob.loc[:, 'var_09'], 
                         y_hat_sigmoid, y_hat_logit, y_hat_inv_logit], axis = 1)

# compute and print classification metrics                    
MCC_metric = matthews_corrcoef(y_predicted.loc[:, 'truth'], y_predicted.loc[:, 'binary'])
precision_metric = precision_score(y_predicted.loc[:, 'truth'], y_predicted.loc[:, 'binary'])
recall_metric = recall_score(y_predicted.loc[:, 'truth'], y_predicted.loc[:, 'binary'])
print('MCC = %.3f\n precision = %.3f\n recall = %.3f' % \
          (MCC_metric, precision_metric, recall_metric ))  


# shows execution time
print( time.time() - start_time, "seconds")
