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
0.6547385707934685

Confusion Matrix
	                Predicted high_risk	Predicted low_risk
Actual high_risk	73	                 28
Actual low_risk	        7069	                 10035

Imbalanced Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.72      0.59      0.02      0.65      0.43       101
   low_risk       1.00      0.59      0.72      0.74      0.65      0.42     17104
```
The balanced accuracy score is 65%.<br>
The high_risk precision is 1% and 72% sensitivity. F1 of 2%.<br>
Low_risk precision is 100% with a sensitivity of 59% and F1 of 74%. This is due to the high numbers in the low risk population. <br>

<br> The **SMOTE algorithms** results are:
```
balanced_accuracy_score(y_test, y_pred)
0.66201409663885

Confusion Matrix
	                Predicted high_risk	Predicted low_risk
Actual high_risk	64	                 37
Actual low_risk	        5296	                 11808

Imbalanced Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.63      0.69      0.02      0.66      0.43       101
   low_risk       1.00      0.69      0.63      0.82      0.66      0.44     17104

avg / total       0.99      0.69      0.63      0.81      0.66      0.44     17205
```
The balanced accuracy score stays close at 66%.<br>
The high_risk precision again is 1% but sensitivity drops down to 63%. F1 is still 2%.<br>
Low_risk precision is 100% with a sensitivity now of 69% and F1 of 63%.  <br>

### Undersampled
**ClusterCentroids algorithm**
```
balanced_accuracy_score(y_test, y_pred)
0.5442369453268994

Confusion Matrix
	                Predicted high_risk	Predicted low_risk
Actual high_risk	70	                 31
Actual low_risk	        10341	                 6763

Imbalanced Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.69      0.40      0.01      0.52      0.28       101
   low_risk       1.00      0.40      0.69      0.57      0.52      0.27     17104

avg / total       0.99      0.40      0.69      0.56      0.52      0.27     17205
```
The balanced accuracy score drops to 54%.<br>
The high_risk precision is 1% sensitivity at 69%. F1 is 1%.<br>
Low_risk precision is 100% with a sensitivity now of 40% and F1 of 57%.  <br>
Undersampled seems to be less accurate than SMOTE or Oversampled

### Combinatorial approach
**SMOTEENN algorithm**
```
balanced_accuracy_score(y_test, y_pred)
0.6472841741611017

Confusion Matrix
	                Predicted high_risk	Predicted low_risk
Actual high_risk	73	                 28
Actual low_risk	        7324	                 9780


Imbalanced Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.72      0.57      0.02      0.64      0.42       101
   low_risk       1.00      0.57      0.72      0.73      0.64      0.41     17104

avg / total       0.99      0.57      0.72      0.72      0.64      0.41     17205
```
The balanced accuracy score drops to 54%.<br>
The high_risk precision is 1% sensitivity at 72%. F1 is 2%.<br>
Low_risk precision is 100% with a sensitivity of 57%% and F1 of 73%. <br> 
No real difference with SMOTEENN and the other models <br>

### Reduced bias models
***Balanced Random Forest Classifier***
```
balanced_accuracy_score(y_test, y_pred)
0.7723403245375988

Confusion Matrix
	                Predicted high_risk	Predicted low_risk
Actual high_risk	67	                 34
Actual low_risk	        2030	                 15074

Imbalanced Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.03      0.66      0.88      0.06      0.76      0.57       101
   low_risk       1.00      0.88      0.66      0.94      0.76      0.60     17104

avg / total       0.99      0.88      0.66      0.93      0.76      0.60     17205
```
The balanced accuracy score is 77%.<br>
The high_risk precision is 3% sensitivity at 66%. F1 is 6%.<br>
Low_risk precision is 100% with a sensitivity of 88%% and F1 of 94%.  <br>
Balanced Random Forest Classifier does significantly better than the previous four models.

***Easy Ensemble Classifier***
```
balanced_accuracy_score(y_test, y_pred)
0.9316600714093861

Confusion Matrix
	                Predicted high_risk	Predicted low_risk
Actual high_risk	93	                 8
Actual low_risk	        983	                 16121


Imbalanced Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.09      0.92      0.94      0.16      0.93      0.87       101
   low_risk       1.00      0.94      0.92      0.97      0.93      0.87     17104

avg / total       0.99      0.94      0.92      0.97      0.93      0.87     17205
```
The balanced accuracy score is the highest so far at 93%.<br>
The high_risk precision is 9% sensitivity at 92%. F1 is 16%.<br>
Low_risk precision is 100% with a sensitivity of 94%% and F1 of 97%.  <br>
Easy Ensemble Classifier exceeds all of the other models performance.<br>

## Summary
When we oversample or undersample or combine over and under sampling our results are not very accurate as can be seen in the accuracy scores of 54 to 66%.  But using ensemble learning with the Balanced Random Forest and Easy Ensemble Classifier models we see that the idea of combining weak learners together can provide more accurate predictions. <br>
<br>Therefore I would recommend that for analyzing credit risk companies would want to consider Ensemble Learning models.
