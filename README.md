# Credit Risk ML Classifier: Data Science Project

![credit-counting image](docs/credit-image.png)

## Dataset

The dataset for this project can be found on [Link](https://rakamin-lms.s3.ap-southeast-1.amazonaws.com/vix-assets/idx-partners/loan_data_2007_2014.csv).

## Objectives

The main objective of this project is:

> **To develop a system that will be able to classify credit risk of a borrower based on the available data and predict whether a borrower has a high or low credit risk by assigning a grade**

To achieve this objective, it was further broken down into the following 3 technical sub-objectives:

1. To perform in-depth exploratory data analysis of the datasets (tabular and graph)
2. To develop a supervised model to classify borrower's credit risk
3. To evaluate and recommend the best model to used

## Main Insights

From the exploratory data analysis, we found out few insight about credit risk patterns:

* There a correlation between loan amount and credit risk. Borrowers with higher credit risk, as indicated by lower credit grades, tend to borrow larger amounts of loans.
* There's orrelation between Interest rate and credit risk. Borrowers with higher credit risk, as indicated by lower credit grades, tend to borrow with higher interest rate, while borrowers with lower credit risk, indicated by higher credit grades, tend to borrow with smaller interest rate.
* Individuals with longer employment lengths tend to demonstrate greater financial stability and reliability, leading to lower credit risk. On the other hand, those with shorter employment lengths might be perceived as less stable, potentially resulting in higher credit risk.
*The higher individual using their available credit, the lower their credit risk grade are. And the lower individual utilized their credit, the higher their grade are.


## Model Selection

Models were compared between each other using their accuracy score.
4 models which are KNeighbors Classifier, Random Forest, Logistic Regression, Decision Tree. 

| Model          | Accuracy |
|----------------|----------|
| KNeighbor      | 0.7      |
| Logistic       | 0.8      |
| Random Forest  | 0.95     |
| Decision Tree  | 0.97     |

The best performing model is Decision Tree Classifier with the following parameters:

```json
{

	ccp_alpha: 0.0, 
	class_weight: None, 
	criterion: gini, 
	max_depth: None, 
	max_features: None, 
	max_leaf_nodes: None, 
	min_impurity_decrease: 0.0, 
	min_samples_leaf: 1, 
	min_samples_split: 2, 
	min_weight_fraction_leaf: 0.0, 
	random_state: None, 
	splitter: 'best'
    
}
```


### Evaluate the model


|             | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Original    |           |        |          |         |
| A           | 1.00      | 1.00   | 1.00     | 13895   |
| B           | 0.99      | 0.99   | 0.99     | 27503   |
| C           | 0.97      | 0.97   | 0.97     | 26254   |
| D           | 0.95      | 0.94   | 0.95     | 15883   |
| E           | 0.94      | 0.98   | 0.96     | 7345    |
| F           | 0.98      | 0.97   | 0.97     | 2723    |
| G           | 0.90      | 0.94   | 0.92     | 610     |
| Accuracy    |           |        | 0.97     | 94213   |
| Macro Avg   | 0.96      | 0.97   | 0.96     | 94213   |
| Weighted Avg| 0.97      | 0.97   | 0.97     | 94213   |

 

1. Feature Importance: The selected model exhibits a well-balanced feature importance distribution. The top 3 features contributing to the model's predictions are X, Y, and Z.
2. Intuition of Feature Importance: The directions of SHAP values align with our expectations. Anomalies are expected to have a larger rate of X and Y and a smaller number of Z, which is consistent with the feature importance findings.
3. Effectiveness of Feature Engineering: The engineered features also demonstrate significance, ranking in the 4th, 5th, and 7th places in terms of importance. This suggests that the feature engineering effort was successful in enhancing the model's performance.

The model achieves high precision, recall, and F1-score across all classes, indicating its effectiveness in classifying each category. The overall accuracy of the model is 0.97, demonstrating strong predictive performance.

## Authors

* [Naufal Fauzan](https://github.com/andnaufal)
