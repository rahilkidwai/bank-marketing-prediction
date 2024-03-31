# Predicting Campaign Marketing Result

- [Predicting Campaign Marketing Result](#predicting-campaign-marketing-result)
  - [Problem Statement](#problem-statement)
    - [Introduction](#introduction)
  - [Dataset](#dataset)
    - [Full Dataset](#full-dataset)
    - [Limited Dataset](#limited-dataset)
    - [Data Attributes](#data-attributes)
    - [Missing Values](#missing-values)
  - [Data Preprocessing](#data-preprocessing)
  - [Techniques / Evaluation](#techniques--evaluation)
    - [Methodology](#methodology)
    - [Results](#results)
      - [Best Performance](#best-performance)
  - [Conclusion](#conclusion)
  - [Future work](#future-work)
  - [References](#references)

## Problem Statement
Compare the performance of the classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines) for multiple marketing campaigns conducted by a Portuguese banking institution. The goal is to predict if the client will subscribe a bank term deposit. The related information and dataset is available [here](https://archive.ics.uci.edu/dataset/222/bank+marketing) [[1]](#1).

### Introduction

## Dataset
### Full Dataset
- Record Count: 41188
- Input Attributes: 20
- Output Attributes: 1
### Limited Dataset
- Record Count: 4119 (10% or total records randomly selected from full dataset)
- Input Attributes: 20
- Output Attributes: 1
### Data Attributes
| Attribute | Type | Input / Output | Description | Unique Values |
| --------- | ---- | ----------- | -------------- | ------------- |
| age | numeric | Input | Age of person | |
| job | categorical | Input | type of job | blue-collar, services, admin., entrepreneur, self-employed, technician, management, student, retired, housemaid, unemployed, unknown |
| marital | categorical | Input | marital status | married, single, divorced, unknown | 
| education | categorical | Input | education level | basic.9y, high.school, university.degree, professional.course, basic.6y, basic.4y, unknown, illiterate | 
| default | categorical | Input | has credit in default? | yes, no, unknown | 
| housing | categorical | Input | has housing loan? | yes, no, unknown | 
| loan | categorical | Input | has personal loan? | yes, no, unknown | 
| contact | categorical | Input | contact communication type | cellular, telephone | 
| month | categorical | Input | last contact month of year | may, jun, nov, sep, jul, aug, mar, oct, apr, dec | 
| day_of_week | categorical | Input | last contact duration, in seconds | fri, wed, mon, thu, tue | 
| duration | numeric | Input | duration of last call | |
| campaign | numeric | Input | number of contacts performed during this campaign | |
| pdays | numeric | Input | number of days that passed by after the client was last contacted from a previous campaign, 999 if not contacted previously | |
| previous | numeric | Input | number of contacts performed before this campaign | | 
| poutcome | categorical | Input | outcome of the previous marketing campaign | nonexistent, failure, success | 
| emp.var.rate | numeric | Input | quarterly indicator of employment variation rate | | 
| cons.price.idx | numeric | Input | monthly indicator of consumer price index | | 
| cons.conf.idx | numeric | Input | monthly indicator of consumer confidence index | | 
| euribor3m | numeric | Input | daily indicator of euribor 3 month rate | | 
| nr.employed | numeric | Input | quarterly indicator of number of employees | | 
| y | categorical | Output | has the client subscribed a term deposit? | yes, no |
### Missing Values
All categorical values coded with the "unknown" are missing values. 

## Data Preprocessing
- Used the limited version of the dataset, which is randomly selected from the full version and is 10% of the full version.
- **Duration Attribute:** The value is not known before a call is performed and is not included for prediction modeling.
- The target value (y) is unevenly distributed with 12% yes and 88% no.
  ![Target Distribution](/images/target_y_dist.png "Target Distribution")
- The heatmap between numerical values
  ![Heatmap Numerical Data](/images/heatmap-numerical-features.png "Heatmap Numerical Data")
## Techniques / Evaluation
Following technology stack is leveraged for this project:
<pre>
Languages: Python
Python Packages: Pandas, nltk, sklearn, matplotlib, seaborn
Notebook: [notebook](notebook.ipynb)
</pre>

### Methodology
This is a Binary Classification problem. Following techniques / models are evaluated:
- KNeighbor Classifier, 
- Logistic Regression,
- Support Vector Machines (SVM) 
- Decision Tree Classifier

There were no missing values in the given dataset. The attribute "duration" is dropped from the dataset. The data was split in Training and Test sets at a factor of 80/20 respectively. The categorical columns were transformed and numerical columns were scaled to have a standardized distribution. 

Models were evaluated with different hyper-parameters and evaluated for accuracy, precision, recall and f-score.

### Results

| Model | Fit Time | Accuracy | Precision | Recall | F1-Score | Train Accuracy |
| ----- | -------- | -------- | --------- | ------ | -------- | -------------- |
| LogisticRegression | 2.50	| 0.9 | 0.88 | 0.9 | 0.87 | 0.91 | 
| KNeighborsClassifier | 34.20 | 0.9 | 0.90 | 0.9 | 0.86 | 0.91 |
| DecisionTreeClassifier | 1.10 | 0.9 | 0.87 | 0.9 | 0.87 | 0.90 |
| SVC | 11.48 | 0.9 | 0.88 | 0.9 | 0.87 | 0.90 |

#### Best Performance
- All of the models achieve an accuracy of 90%
- Best precision of 90% is achieved by KNeighbors Classifier followed by Logistic Regression and SVC
- All of the models achieve a recall score of 90%
- Best F1-Score of 87% is achieved by Logistic Regression, Decision Tree and SVC
- Best fit time was of Decision Classifier followed by Logistic regression
- Overall any one of Logistics Regression or SVC can be chosen to predict future values as they are more performant compared to KNeighbors classifier which will need complete dataset to make the prediction.
  ![Logistic Regression](/images/cm_lr.png "Logistic Regression") ![SVC](/images/cm_svc.png "SVC")

## Conclusion
To predict the code, we explored four different ML techniques:
- KNeighbor Classifier, 
- Logistic Regression,
- State Vector Machines (SVM) 
- Decision Tree Classifier

For comparison we used the following metrics:
- Accuracy, precision, recall, f-score and looked at average fit time to compare performance.

Here are the ranges of values for all parameters based on our evaluation:
| Parameter | Range |
| --------- | ----- |
| Accuracy | 0.90 |
| Precision | 0.87 - 0.90 |
| Recall | 0.90 |
| F1-Score | 0.86 - 0.87 |
| Fit time (seconds) | 1.10s - 11.48s |

## Future work
The evaluation was done using the limited dataset (10% of full dataset) because of the resource constraints and computing power limitations. To have more accurate evaluation, it is recommended to perform the evaluation on the full dataset and expanded range of hyper-parameters.

## References
<a id="1">[1]</a> [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, In press, http://dx.doi.org/10.1016/j.dss.2014.03.001