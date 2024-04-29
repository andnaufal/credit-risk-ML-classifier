# Credit Risk ML Classifier: Data Science Project

![credit-counting image](docs/credit-image.jpg)

## Overview

This project aims to develop a machine learning system for classifying credit risk based on borrower data, ultimately predicting whether a borrower has high or low credit risk by assigning a grade. The project focuses on exploratory data analysis (EDA), model development, and evaluation.

### Dataset

The dataset used for this project is available [here](https://rakamin-lms.s3.ap-southeast-1.amazonaws.com/vix-assets/idx-partners/loan_data_2007_2014.csv).

### Objectives

The main objectives of this project are:

- Conduct comprehensive exploratory data analysis (EDA) to understand credit risk patterns.
- Develop and compare different supervised machine learning models to classify borrower credit risk.
- Identify and recommend the best-performing model for credit risk prediction.

### Key Insights

#### Exploratory Data Analysis (EDA)

During EDA, several insights were discovered:

- There's a positive correlation between loan amount and credit risk; borrowers with higher risk tend to borrow larger sums.
- Interest rates correlate with credit risk; riskier borrowers often face higher interest rates.
- Longer employment lengths are associated with lower credit risk, suggesting greater financial stability.
- Credit utilization inversely affects credit risk; borrowers using less of their available credit tend to have higher credit grades.

### Model Selection and Performance

Four models were compared based on accuracy scores:

| Model           | Accuracy |
|-----------------|----------|
| KNeighbors      | 0.70     |
| Logistic Regression | 0.80 |
| Random Forest   | 0.95     |
| Decision Tree   | 0.97     |

The Decision Tree Classifier outperformed other models with an accuracy of 0.97. The optimal parameters for this model include:

```json
{
	"ccp_alpha": 0.0,
	"class_weight": null,
	"criterion": "gini",
	"max_depth": null,
	"max_features": null,
	"max_leaf_nodes": null,
	"min_impurity_decrease": 0.0,
	"min_samples_leaf": 1,
	"min_samples_split": 2,
	"min_weight_fraction_leaf": 0.0,
	"random_state": null,
	"splitter": "best"
}
```

### Model Evaluation

The Decision Tree Classifier achieved high precision, recall, and F1-score across all credit risk categories:

| Category      | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| A             | 1.00      | 1.00   | 1.00     | 13895   |
| B             | 0.99      | 0.99   | 0.99     | 27503   |
| C             | 0.97      | 0.97   | 0.97     | 26254   |
| D             | 0.95      | 0.94   | 0.95     | 15883   |
| E             | 0.94      | 0.98   | 0.96     | 7345    |
| F             | 0.98      | 0.97   | 0.97     | 2723    |
| G             | 0.90      | 0.94   | 0.92     | 610     |
| **Accuracy**  |           |        | **0.97** | 94213   |
| **Macro Avg** | 0.96      | 0.97   | 0.96     | 94213   |
| **Weighted Avg** | 0.97   | 0.97   | 0.97     | 94213   |

The model demonstrates robust performance with an overall accuracy of 0.97.

## Authors

- [Naufal Fauzan](https://github.com/andnaufal)

### How to Use

1. **Installation**:
   - Clone this repository.
   - Install the required dependencies using `pip install -r requirements.txt`.

2. **Usage**:
   - Explore the Jupyter notebooks in the `notebooks/` directory for detailed analysis.
   - Run `python train.py` to train the Decision Tree Classifier and evaluate its performance.

### Future Improvements

- Implement feature engineering techniques to enhance model performance.
- Explore advanced algorithms like gradient boosting for further model improvement.

### Contact

For questions or collaborations, feel free to reach out to [Naufal Fauzan](https://github.com/andnaufal).
