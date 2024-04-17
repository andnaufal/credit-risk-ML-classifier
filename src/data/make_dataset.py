import pandas as pd
data = pd.read_csv("../../data/raw/loan_data_2007_2014.csv")
data.to_pickle("../../data/interim/dataCredit.csv")