# Credit Risk Analysis

## Overview
Good loans easily outnumber risky loans.  Due to the large number of loans made these days machine learning can be used determine credit risk.   Since credit risk is an inherently unbalanced classification problem, meaning either a person gets a loan or not, this analysis will use different techniques to train and evaluate models with the unbalanced classes to determine which model might be best used to predict credit risk.

## Setup
The Python libraries **imbalanced-learn** and **scikit-learn** are used to build and evaluate models using resampling. <br>
The dataset used comes from LendingClub, a peer-to-peer lending service. <br><br>
To predict credit risk the data will be:
* **Oversampled** using:
  * Random Over Sampler and 
  * SMOTE algorithms
* **Undersampled** using:
  * ClusterCentroids algorithm
* A **Combinatorial approach** of over- and undersampling using:
  * SMOTEENN algorithm
* In addition two machine learning models that **reduce bias** will be used:
  * Balanced Random Forest Classifier
  * Easy Ensemble Classifier

## Results
### Oversampled
In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. <br>The **Random Over Sampler** algorithm results are:
```
balanced_accuracy_score(y_test, y_pred)
0.6383950485787587

Confusion Matrix
	                Predicted high_risk	Predicted low_risk
Actual high_risk	73	                 28
Actual low_risk	        7290	                 9814

Imbalanced Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.66      0.61      0.02      0.64      0.41       101
   low_risk       1.00      0.61      0.66      0.76      0.64      0.40     17104

avg / total       0.99      0.61      0.66      0.76      0.64      0.40     17205
```
The balanced accuracy score is 65%.
The high_risk precision is 1% and 66% sensitivity. F1 of 2%.
Low_risk precision is 100% with a sensitivity of 61% and F1 of 76%. This is probably due to the high numbers in the low risk population. 

<br> The **SMOTE algorithms** results are:
```
balanced_accuracy_score(y_test, y_pred)
0.6589052760514592

Confusion Matrix
	                Predicted high_risk	Predicted low_risk
Actual high_risk	73	                 28
Actual low_risk	        7290	                 9814

Imbalanced Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.62      0.69      0.02      0.66      0.43       101
   low_risk       1.00      0.69      0.62      0.82      0.66      0.44     17104

avg / total       0.99      0.69      0.62      0.81      0.66      0.44     17205
```
The balanced accuracy score stays about the same at 65%.
The high_risk precision again is 1% but sensitivity drops down to 62%. F1 is still 2%.
Low_risk precision is 100% with a sensitivity now of 69% and F1 of 62%.  

### Undersampled
**ClusterCentroids algorithm**
```
balanced_accuracy_score(y_test, y_pred)
0.6589052760514592

Confusion Matrix
	                Predicted high_risk	Predicted low_risk
Actual high_risk	73	                 28
Actual low_risk	        7290	                 9814

Imbalanced Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.62      0.69      0.02      0.66      0.43       101
   low_risk       1.00      0.69      0.62      0.82      0.66      0.44     17104

avg / total       0.99      0.69      0.62      0.81      0.66      0.44     17205
```
The balanced accuracy score stays about the same at 65%.
The high_risk precision again is 1% but sensitivity drops down to 62%. F1 is still 2%.
Low_risk precision is 100% with a sensitivity now of 69% and F1 of 62%.  

### Combinatorial approach
**SMOTEENN algorithm**
### Reduced bias models
***Balanced Random Forest Classifier***
***Easy Ensemble Classifier***
