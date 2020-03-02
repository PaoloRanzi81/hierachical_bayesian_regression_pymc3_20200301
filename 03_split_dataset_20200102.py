"""
TITLE: "script for splitting a huge data-set in small chunks"
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
import time



###############################################################################
## 2. SETTING PATHS AND KEYWORDS
# In order to set the correct pathways/folders, check which system are you
# using. It should be either Linux laptop (release == '5.0.0-29-generic') 
# or Linux server (release == '4.4.0-143-generic').
RELEASE = platform.release()

if RELEASE == '5.3.0-40-generic': # Linux laptop
   BASE_DIR_INPUT = ('/media/paolo/4C0F-08B1/analyses/python/tensorflow/pefferkoven_marco/raw_data')
   BASE_DIR_OUTPUT = ('/media/paolo/4C0F-08B1/analyses/python/tensorflow/pefferkoven_marco/outputs')

else:
   BASE_DIR_INPUT = ('/home/ubuntu/raw_data')
   BASE_DIR_OUTPUT = ('/home/ubuntu/outputs')

   
###############################################################################
## 3. PARAMETERS TO BE SET!!!
input_file_name_1 = ('dataset_marketing.csv')
output_file_name_6 = None 
#output_file_name_6 = ('clean_marketing.csv')
#input_file_name_1 = None
number_of_splits = 5 # choose whichever number, depending on your data-set's size

# start clocking time
start_time = time.time()


###############################################################################
## 4. LOADING DATA-SET 
if output_file_name_6 is not None:
    # loading the .csv file 
    X_tmp = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                     output_file_name_6]), header = 0) 
#elif input_file_name_1 is not None: 
else: 
    # loading the .csv file 
    X_tmp = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                 input_file_name_1]), header = 0)

# randomly sampling half of the data-set
X = X_tmp.copy()  

# free-up RAM memory
X_tmp = None

# split into chunks
chunks_df = np.array_split(X, number_of_splits, 0) 

for chunk in range(0, (len(chunks_df))): 
        
    # generate chunk's name
    chunk_name = ("00%s_chunk.csv" % (chunk+1) )
    
    # convert chunk (numpy array) into a Pandas DataFrame
    split_chunk = pd.DataFrame(chunks_df[chunk])
    
    # save chunk as separate csv file: 
    split_chunk.to_csv(os.path.sep.join([BASE_DIR_INPUT, chunk_name]), index = False) 
     

# shows execution time
print( time.time() - start_time, "seconds")
    
    
