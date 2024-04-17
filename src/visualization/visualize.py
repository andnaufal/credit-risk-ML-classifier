import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
import math

data = pd.read_pickle("../../data/processed/data_credit_processed.csv")

import matplotlib.pyplot as plt
import seaborn as sns

#### 1. Distribution of the grades as the dependent variable
# Set plot style
plt.style.use("ggplot")

# Create count plot
plt.figure(figsize=(5, 5))
ax = sns.countplot(x=data["grade"], palette="rocket")
ax.bar_label(ax.containers[0])

# Add title
plt.title('Distribution of the grades as the dependent variable')

# Save the plot
plt.savefig('../../reports/figures/Distribution_of_the_grades_as_the_dependent_variable.png', transparent=False)

# Show the plot
plt.show()

#### 2. Correlation between loan amount and credit risk
# Create the box plot
plt.figure(figsize=(8, 6))
data.boxplot(column='loan_amnt', by='grade', grid=False)
plt.title('Box Plot of Loan Amount by Credit Risk Grade')
plt.xlabel('Credit Risk Grade')
plt.ylabel('Loan Amount')
plt.suptitle('')  # Remove default title

# Save the plot
plt.savefig('../../reports/figures/Box_Plot_of_Loan_Amount_by_Credit_Risk_Grade.png', transparent=False)

# Show the plot
plt.show()

#### 3. Relationship between interest rate and credit risk
# Create the box plot
plt.figure(figsize=(8, 6))
data.boxplot(column='int_rate', by='grade', grid=False)
plt.title('Box Plot of Interest Rate by Credit Risk Grade')
plt.xlabel('Credit Risk Grade')
plt.ylabel('Interest Rate')
plt.suptitle('') 

# Save the plot
plt.savefig('../../reports/figures/Box_Plot_of_Interest_Rate_by_Credit_Risk_Grade.png', transparent=False)

# Show the plot
plt.show()

#### 4. Length of employment
# Filter data for the specified employment length categories
employment_categories = ['10+ years', '< 1 year', '1 year', '3 years', '8 years', '9 years', '4 years', '5 years', '6 years', '2 years', '7 years']
filtered_data = data[data['emp_length'].isin(employment_categories)]

# Group data by 'grade' and 'emp_length', then count the occurrences
grouped_data = filtered_data.groupby(['grade', 'emp_length']).size().unstack(fill_value=0)

colors = sns.color_palette("Paired", len(grouped_data.columns))

# Plot stacked bar chart
grouped_data.plot(kind='bar', stacked=True, figsize=(10, 6),color=colors)

# Customize plot
plt.xlabel('Credit Risk Grade')
plt.ylabel('Count')
plt.title('Credit Risk Distribution by Employment Length')
plt.legend(title='Employment Length')

# Save plot
plt.savefig('../../reports/figures/Credit_Risk_Distribution_by_Employment_Length.png', transparent=False)

# Show plot
plt.show()

# Print the table
print("Count of Employment Length Categories within Each Credit Risk Grade:")
print(grouped_data)

#### 5. Patterns in the types of loans (purpose) and credit risk
# Filter data for the specified employment length categories
purpose = data["purpose"]
filtered_data = data[data['purpose'].isin(purpose)]

# Group data by 'grade' and 'emp_length', then count the occurrences
grouped_data = filtered_data.groupby(['grade', 'purpose']).size().unstack(fill_value=0)

colors = sns.color_palette("Paired", len(grouped_data.columns))

# Plot stacked bar chart
grouped_data.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)

# Customize plot
plt.xlabel('Credit Risk Grade')
plt.ylabel('Count')
plt.title('Credit Risk Distribution by Borrower Purpose')
plt.legend(title='Borrow Purpose')

# Save plot
plt.savefig('../../reports/figures/Credit_Risk_Distribution_by_Borrower_Purpose.png', transparent=False)

# Show plot
plt.show()

#### 6. Relationship between credit utilization (revol_util) and credit risk
# Create the box plot
plt.figure(figsize=(8, 6))
data.boxplot(column='revol_util', by='grade', grid=False)
plt.title('Box Plot of Credit Utilization by Credit Risk Grade')
plt.xlabel('Credit Risk Grade')
plt.ylabel('Credit Utilization')
plt.suptitle('') 

# Save the plot before displaying it
plt.savefig('../../reports/figures/Box_Plot_of_Credit_Utilization_by_Credit_Risk_Grade.png', transparent=False)

# Show the plot
plt.show()

# Calculate median loan amount for each grade
median_loan_amounts = data.groupby('grade')['revol_util'].median().reset_index()
median_loan_amounts.columns = ['Credit Risk Grade', 'Credit Utilization']

# Print the table
print("Median Credit Utilization by Credit Risk Grade:")
print(median_loan_amounts)
