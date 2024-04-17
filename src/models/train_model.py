import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression, LogisticRegression
import math
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_pickle("../../data/processed/data_credit_processed.csv")

y = data['grade']
x = data.drop('grade', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size = .75) #puts 75 percent of the data into a training set and the remaining 25 percent into a testing set.
print('Shape of x_train and y_train: ',x_train.shape, y_train.shape)
print('Shape of x_test and y_test: ',x_test.shape, y_test.shape)

subsample_size = 0.5

x_train_dum = pd.get_dummies(x_train[['emp_length','verification_status','loan_status','purpose','initial_list_status','application_type']])
x_test_dum = pd.get_dummies(x_test[['emp_length','verification_status','loan_status','purpose','initial_list_status','application_type']])


x_train = pd.concat([x_train, x_train_dum], axis = 1)
x_test = pd.concat([x_test, x_test_dum], axis = 1)

x_train.drop(['emp_length','verification_status','loan_status','purpose','initial_list_status','application_type'], axis = 1, inplace = True)
x_test.drop(['emp_length','verification_status','loan_status','purpose','initial_list_status','application_type'], axis = 1, inplace = True)

columns_to_keep = ['int_rate', 'total_rec_int', 'installment', 'total_rec_prncp', 
                   'term', 'last_pymnt_amnt', 'revol_util', 'total_pymnt', 
                   'total_pymnt_inv', 'loan_amnt', 'funded_amnt', 'total_rev_hi_lim', 
                   'dti', 'tot_cur_bal']

# List of columns to drop
columns_to_drop = [col for col in x_train.columns if col not in columns_to_keep]
columns_to_drop2 = [col for col in x_test.columns if col not in columns_to_keep]

# Drop columns
x_train.drop(columns=columns_to_drop, inplace=True)
x_test.drop(columns=columns_to_drop2, inplace=True)

# Import the necessary libraries
from imblearn.over_sampling import SMOTE

# Creating an instance of SMOTE
smote = SMOTE()

# Balancing the data
x_train, y_train = smote.fit_resample(x_train, y_train)






models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "KNeighbors Classifier": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier()
}

for model_name, model in models.items():
    model.fit(x_train, y_train)
    
    # Calculate accuracy score
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy score for {model_name}: {accuracy:.2f}")
    

import os

# Define the directory to save the models
save_dir = "../../src/models"

# Create the directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize variables to track the highest accuracy and the corresponding model name
best_accuracy = 0
best_model_name = ""

# Loop through each model
for model_name, model in models.items():
    # Evaluate the model's accuracy
    accuracy = model.score(x_test, y_test)  # Replace x_test and y_test with your test data
    # Check if the current model has higher accuracy than the previous best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name

# Get the best model
best_model = models[best_model_name]

# Define the filename for the best model
filename = os.path.join(save_dir, f"{best_model_name.lower().replace(' ', '_')}_model.pkl")

# Save the best model
with open(filename, 'wb') as file:
    pickle.dump(best_model, file)
