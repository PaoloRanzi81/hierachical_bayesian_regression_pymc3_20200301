# hierachical_bayesian_regression_pymc3_20200301
Hierachical Bayesian regression by PyMC3 + prediction performance evaluation 

PROJECT NAME: 'hierachical_bayesian_model_regression_20200102'
AUTHOR: Paolo Ranzi
README FILE VERSION (LAST UPDATE): 20200301


Python 3.6.7 has been used. For each step the specific Python script has been mentioned, accordingly. At the begginning of each script we have to make sure of setting custom-made parameters/pieces of information: 
- import Python libraries (if you do not have a specific library you have to manually install it by using PIP within your virtual enviroment);  
- setting paths and keywords according to your storage location;
- set parameters (e.g. input file name, output file name etc.); 
- all scripts have been optimized for using them by parallel computing. Please set number of CPUs by searching for 'cpu_count()' within each script according to your available resources; 
- SYNOPSYS: script 01 + 02 corresponds at not hierachical logistic regression (used just for comparison
with Hierachical Bayesian Model (HBM)); instead script 03 + 04 + 05 have been used for the more interesting HBM logistic regression;  

STEPS: 

01. LEARNING STEP (NOT HIERACHICAL)
(it computes Bayesian + MCMC not-hierachical logistic regression by PyMC3): 
SCRIPT NAME: '01_logistic_regression_20200102.py'
INPUT: .csv file; 
OUTPUT: out-of-bag .csv file; pickeled bayesian model ( .joblib); pickeled MCMC traces (to be used later for diagnositics and prediction);

02. VALIDATION STEP BY CHECKING ALGORITHM'S PREDICTIIVE PERFORMANCE: 
(it computes MCC + precision + recall metrics on the out-of-bag data-set):
SCRIPT NAME: '02_logistic_prediction_20200102.py'
INPUT: '01_logistic_regression_20200102.py' + out-of-bag .csv file;
OUTPUT: model validation metrics (i.e. MCC, precision and reacall); 

03. SPLITTING DATA-SET IN CHUNKS
(it computes Bayesian + MCMC not-hierachical logistic regression by PyMC3): 
SCRIPT NAME: '03_split_dataset_20200102.py'
INPUT: .csv file; 
OUTPUT: .csv files, one for each chunk;

04. LEARNING STEP (HIERACHICAL)
(it computes Bayesian + MCMC hierachical logistic regression by PyMC3): 
SCRIPT NAME: '04_HBM_analysis_20200102.py'
INPUT: .csv files, one for each chunk; 
OUTPUT: out-of-bag .csv files; pickeled bayesian model ( .joblib); pickeled MCMC traces (to be used later for diagnositics and prediction);

05. VALIDATION STEP BY CHECKING ALGORITHM'S PREDICTIIVE PERFORMANCE: 
(it computes MCC + precision + recall metrics on the out-of-bag data-set):
SCRIPT NAME: '05_HBM_prediction_20200102.py'
INPUT: '04_HBM_analysis_20200102.py' + out-of-bag .csv files;
OUTPUT: model validation metrics (i.e. MCC, precision and reacall);









