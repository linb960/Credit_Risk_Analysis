# Credit Risk Analysis

## Overview
Good loans easily outnumber risky loans.  Due to the large number of loans made these days machine learning can be used determine credit risk.   Since credit risk is an inherently unbalanced classification problem, meaning either a person gets a loan or not, this analysis will use different techniques to train and evaluate models with the unbalanced classes to determine which model might be best used to predict credit risk.

## Setup
The Python libraries **imbalanced-learn** and **scikit-learn** are used to build and evaluate models using resampling. <br>
The dataset used comes from LendingClub, a peer-to-peer lending service. <br><br>
To predict credit risk the data will be:
* **Oversampled** using:
  * RandomOverSampler
  *  SMOTE algorithms
* **Undersampled** using:
  * ClusterCentroids algorithm
* A **Combinatorial approach** of over- and undersampling using:
  * SMOTEENN algorithm
* In addition two machine learning models that **reduce bias** will be used:
  * BalancedRandomForestClassifier
  * EasyEnsembleClassifier
