# Predicting Campaign Marketing Result

- [Predicting Campaign Marketing Result](#predicting-campaign-marketing-result)
  - [Problem Statement](#problem-statement)
    - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Data Preprocessing](#data-preprocessing)
    - [DataFrame for Model Evaluation](#dataframe-for-model-evaluation)
  - [Techniques / Evaluation](#techniques--evaluation)
    - [1. Logistic Regression](#1-logistic-regression)
    - [2. KNeighbor Classifier](#2-kneighbor-classifier)
    - [3. Support Vector Machines](#3-support-vector-machines)
    - [4. Decision Tree Classifier](#4-decision-tree-classifier)
  - [Conclusion](#conclusion)
  - [References](#references)

## Problem Statement


### Introduction

## Dataset

## Data Preprocessing

### DataFrame for Model Evaluation
Stats on notes related to diabetes diagnosis:
|   |   |
|---|---|
| Total count | 25,376 |


## Techniques / Evaluation
Following technology stack is leveraged for this project:
<pre>
Languages: Python
Python Packages: Pandas, nltk, sklearn, matplotlib, seaborn
Notebook: [notebook](notebook.ipynb)
</pre>

This is a Multi Label Classification problem. I explored following Machine Learning techniques / models:
- KNeighbor Classifier, 
- Logistic Regression,
- State Vector Machines (SVM) 
- Decision Tree Classifier \

To compare different techniques I use the following metrics: Accuracy, and F-score (Macro and Weighted).


### 1. Logistic Regression

| Multi Class | Fit Time | Accuracy | F1-Score (macro) | F1-Score (weighted) |
| ----------- | -------- | -------- | ---------------- | ------------------- |
| One-vs-Rest | 3.92s | 0.8395 |	0.7056 | 0.7965 |
| Multinomial | 3.08s |	0.8395 | 0.7056 | 0.7965 |
| One-vs-One | 17.11s | 0.8519 | 0.7395 | 0.8225 |
| One-vs-Rest | 2.96s | 0.7778 | 0.6357 | 0.7339 |
| Multinomial | 2.25s | 0.7901 | 0.6573 | 0.7501 |
| One-vs-One | 12.85s | 0.6790 | 0.5060 | 0.6267 |

**Best Performance**:

### 2. KNeighbor Classifier
| Vectorizer | Fit Time | Accuracy | F1-Score (macro) | F1-Score (weighted) |
| --------- | -------- | -------- | ---------------- | ------------------- |
|  | 0.48s | 0.7778 | 0.6713 | 0.7479 |
|  | 0.58s | 0.6543 | 0.4809 | 0.6031 |

**Best Performance**: 

### 3. Support Vector Machines
| Kernel | Fit Time | Accuracy | F1-Score (macro) | F1-Score (weighted) |
| ------ | ------- | --------- | ---------------- | ------------------- |
| RBF | 1.15s | 0.1975 | 0.0300 | 0.0652 |
| Poly | 1.24s | 0.5556 | 0.4647 | 0.5066 |
| RBF | 1.28s | 0.1975 | 0.0300 | 0.0652 |
| Poly | 1.39s | 0.7160 | 0.5839 | 0.6786 |

**Best Performance**: 

### 4. Decision Tree Classifier
| Multi Class | Fit Time | Accuracy | F1-Score (macro) | F1-Score (weighted) |
| ----------- | -------- | -------- | ---------------- | ------------------- |
| OVR | 468.94s | 0.8025 | 0.6602 | 0.7642 |
| Multinomial | 402.24s | 0.7654 | 0.6591 | 0.7392 |
| OVR | 264.45s | 0.8025 | 0.6602 | 0.7642 |
| Multinomial | 211.76s | 0.7654 | 0.6560 | 0.7413 |

**Best Performance**:

## Conclusion
To predict the code, we explored four different ML techniques:
- KNeighbor Classifier, 
- Logistic Regression,
- State Vector Machines (SVM) 
- Decision Tree Classifier

For comparison we used the following metrics:
- Accuracy
- F-score (macro / weighted)
And looked at average fit time to compare performance.

Here are the ranges of values for all parameters based on our evaluation:
| Parameter | Range |
| --------- | ----- |
| Accuracy | 0.1975 - 0.8519 |
| F-score (macro) | 0.0300 - 0.7395 |
| F-score (weighted) | 0.5066 - 0.8225 |
| Fit time (seconds) | 0.48s - 468.94s |

**Overall Best Performance**:
| Parameter | Best Model | Vectorizer |
| --------- | ---------- | ------------- |
| Accuracy | Logistic Regression (class = ovo) | Count Vectorizer |
| F-score (macro) | Logistic Regression (class = ovo) | Count Vectorizer |
| F-score (weighted) | Logistic Regression (class = ovo) | Count Vectorizer |
| Fit time (seconds) | Naive Bayes | Count Vectorizer | 

**Overall Worst Performance**:
| Parameter | Worst Model | Vectorizer |
| --------- | ---------- | ------------- |
| Accuracy | SVM (kernel = rbf) | Count Vectorizer & TF-IDF |
| F-score (macro) | SVM (kernel = rbf) | Count Vectorizer & TF-IDF |
| F-score (weighted) | SVM (kernel = poly) | Count Vectorizer |
| Fit time (seconds) | AdaBoost (class = ovr) | Count Vectorizer |


## References
<a id="1">[1]</a> https://archive.ics.uci.edu/dataset/222/bank+marketing