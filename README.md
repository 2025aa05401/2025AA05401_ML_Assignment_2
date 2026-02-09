# 2025AA05401_ML_Assignment_2

Problem Statement :-
The objective of this assignment is to build and evaluate multiple machine learning classification models for predicting whether an individual’s annual income exceeds $50K based on demographic and employment-related attributes. The task involves implementing six different classification algorithms on the same dataset and comparing their performance using multiple evaluation metrics to identify the most effective model for this classification problem.

Dataset Description :-

Dataset Name: Adult Income Dataset

Source: UCI Machine Learning Repository

Problem Type: Binary Classification

Dataset Characteristics:

Number of instances: 48,842

Number of features: 14 (after preprocessing)

Target variable:

<=50K → Income less than or equal to $50K
>50K → Income greater than $50K

Feature types: Combination of categorical and numerical features such as age, education, occupation, workclass, hours-per-week, etc.

Missing values: Removed during preprocessing

The following six machine learning models were implemented on the Adult Income dataset: Logistic Regression, Decision Tree Classifier, K-Nearest Neighbors (kNN),Naive Bayes (Gaussian),Random Forest (Ensemble Model),XGBoost (Ensemble Model). Each model was evaluated using the following metrics:Accuracy,AUC (Area Under the ROC Curve),Precision,
Recall,F1 Score,Matthews Correlation Coefficient (MCC)

| ML Model Name            | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC      |
| ------------------------ | -------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression      | 0.827883 | 0.860793 | 0.724623  | 0.459821 | 0.562622 | 0.480593 |
| Decision Tree            | 0.812529 | 0.753214 | 0.604707  | 0.639031 | 0.621395 | 0.497277 |
| kNN                      | 0.834024 | 0.856943 | 0.671117  | 0.609056 | 0.638582 | 0.532241 |
| Naive Bayes              | 0.808076 | 0.864383 | 0.704370  | 0.349490 | 0.467178 | 0.399403 |
| Random Forest (Ensemble) | 0.859205 | 0.910814 | 0.740576  | 0.639031 | 0.686066 | 0.598644 |
| XGBoost (Ensemble)       | 0.876248 | 0.928644 | 0.776087  | 0.683036 | 0.726594 | 0.649242 |

| ML Model Name            | Observation about Model Performance                                                                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Achieved good accuracy and AUC but had low recall, indicating difficulty in correctly identifying high-income individuals due to class imbalance and linear decision boundaries. |
| Decision Tree            | Showed balanced precision and recall but lower AUC, suggesting overfitting and limited generalization capability.                                                                |
| kNN                      | Performed better than single-tree models with improved F1 score and MCC, but performance depended heavily on feature scaling and the choice of neighborhood size.                |
| Naive Bayes              | Achieved relatively high AUC but very low recall, indicating strong probabilistic separation but poor detection of the minority (>50K) class due to the independence assumption. |
| Random Forest (Ensemble) | Significantly improved all evaluation metrics by combining multiple decision trees, effectively capturing non-linear relationships and reducing overfitting.                     |
| XGBoost (Ensemble)       | Delivered the best overall performance across all metrics. Its boosting approach handled class imbalance and complex feature interactions, making it the most reliable model.    |




