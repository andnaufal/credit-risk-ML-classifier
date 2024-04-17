import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import StandardScaler

# Read the CSV file with the specified encoding
data = pd.read_pickle("../../data/interim/dataCredit.csv")  # or encoding='ISO-8859-1'

data=data.drop(['inq_fi', 'total_cu_tl', 'inq_last_12m','open_acc_6m', 'open_il_6m', 'open_il_12m',
       'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
       'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util','annual_inc_joint', 'dti_joint',
       'verification_status_joint','Unnamed: 0', 'id', 'member_id','desc','pymnt_plan','zip_code','url','delinq_2yrs','last_pymnt_d','next_pymnt_d','last_credit_pull_d','earliest_cr_line','issue_d','emp_title','title','addr_state','sub_grade','home_ownership'], axis=1)
data = data.dropna(subset=['emp_length','annual_inc', 'inq_last_6mths','pub_rec', 'revol_bal', 'revol_util', 'total_acc','last_pymnt_amnt',
       'collections_12_mths_ex_med','acc_now_delinq','tot_coll_amt','total_rev_hi_lim','tot_cur_bal']) 
data['mths_since_last_delinq'] = data['mths_since_last_delinq'].fillna(data['mths_since_last_delinq'].mean())
data['mths_since_last_record'] = data['mths_since_last_record'].fillna(data['mths_since_last_record'].mean())
data['mths_since_last_major_derog'] = data['mths_since_last_major_derog'].fillna(data['mths_since_last_major_derog'].mean())
data['term'] = data['term'].str.replace('months', '')
data['term'] = pd.to_numeric(data['term'])
data['initial_list_status'] = data['initial_list_status'].str.replace('w', 'whole_loan')
data['initial_list_status'] = data['initial_list_status'].str.replace('f', 'fractional_loan')
from sklearn import preprocessing

# Select numerical features for normalization
numeric_features = data.select_dtypes(include=['int', 'float']).columns

# Normalization
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

data.to_pickle("../../data/processed/data_credit_processed.csv")