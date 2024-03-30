# Predicting ICD Diagnosis Codes from EHR Records
<center>
    <center>
        <img src=images/med-billing.jpg width=50% />
    </center>
</center>

- [Predicting ICD Diagnosis Codes from EHR Records](#predicting-icd-diagnosis-codes-from-ehr-records)
  - [Problem Statement](#problem-statement)
    - [Background](#background)
  - [Business Value](#business-value)
  - [Dataset](#dataset)
    - [MIMIC-IV](#mimic-iv)
    - [Data Structure](#data-structure)
    - [Data Loading](#data-loading)
      - [Data Format](#data-format)
      - [Data Exploration](#data-exploration)
  - [Data Preprocessing](#data-preprocessing)
    - [Study Constraints](#study-constraints)
    - [New Schema](#new-schema)
    - [DataFrame for Model Evaluation](#dataframe-for-model-evaluation)
      - [DataCleaner utility](#datacleaner-utility)
      - [Final Dataset](#final-dataset)
  - [Techniques / Evaluation](#techniques--evaluation)
    - [Text Vectorization](#text-vectorization)
    - [1. Logistic Regression](#1-logistic-regression)
    - [2. Naive Bayes](#2-naive-bayes)
    - [3. Support Vector Machines](#3-support-vector-machines)
    - [4. AdaBoost Ensemble](#4-adaboost-ensemble)
  - [Future Work](#future-work)
  - [References](#references)

## Problem Statement
Automate and recommend diagnosis and procedure codes based on discharge summaries and clinical notes in healthcare domain.

### Introduction
The International Classification of Diseases (ICD) code is a diagnostic classification standard that is frequently used as a referencing system in healthcare and insurance. \
ICD-10 (International Classification of Diseases, 10th Revision) is a set of codes used to classify diseases, injuries, health encounters, and inpatient procedures published by the World Health Organization (WHO). ICD essentially categorizes diseases and has given each illness a unique code. The ICD-10 version of codes contains more than 70,000 codes. \

ICD codes have been broadly adopted by healthcare providers for healthcare claims reimbursement and billing. Thousands of medical coders do the assginment (classification) manually by reading patient health records, physician notes and discharge summaries and extractng key information to determine and assign correct codes. The complex structure and amount of information in Electronic Medicial Record (EMR) systems significantly increase the difficulty of manual coding which is also time consuming and error prone. Billing issues and underpayments can result from coding errors and missing codes.

Automation of coding have become popular in recent years because of the large scale adoption of machine learing and developments in artificial intelligence like deep learning via neural networks and natural language processing (NLP) models. 

Any reduction in the laborious task of manual coding could assist physicians in making a better diagnosis, and reduce billing and claim errors. Also prediction can help in the completeness (predict missing diagnosis from EHR) of diagnosis.

## Dataset
Electronic Medical Record system (EMR) is a great source of text data as it includes clinical notes, discharge notes, assessments, current medications, and ICD-10 codes.

MIMIC (Medical Information Mart for Intensive Care) [[1]](#1) [[2]](#2) [[3]](#3) is a large, freely-available database comprising de-identified health-related data associated with over forty thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012. \
The database includes information such as demographics, vital sign measurements, laboratory test results, procedures, medications, caregiver notes, imaging reports.

### MIMIC-IV
MIMIC IV is the latest release of the dataset. To get access to the dataset, one has to agree to the Data Use Agreement and complete the required training. \
[MIMIC-IV Version 2.2](https://physionet.org/content/mimiciv/2.2/) has been used for this project. \
**Access Policy:** Only credentialed users who sign the DUA can access the files. \
**License (for files):** [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mimiciv/view-license/2.2/) \
**Data Use Agreement:** [PhysioNet Credentialed Health Data Use Agreement 1.5.0](https://physionet.org/content/mimiciv/view-dua/2.2/) \
**Required training:** [CITI Data or Specimens Only Research](https://physionet.org/content/mimiciv/view-required-training/2.2/#1)
### Data Structure
MIMIC-IV dataset comprises of different modules as described below: \
**Hosp** module contains all data acquired from the hospital wide electronic health record. Information covered includes patient and admission information, laboratory measurements, microbiology, medication administration, and billed diagnoses. \
**ICU** module contains information collected from the clinical information system used within the ICU and includes intravenous administrations, ventilator settings, and other charted items. \
**ED** module contains data for emergency department patients and includes reason for admission, triage assessment, vital signs, and medicine reconciliation. \
**CXR** module contains patient chest x-rays and can linked with the clinical data from other modules. \
**ECG** module contains waveform data, and lookup tables which can be used to link subjects to other modules. \ 
**Note** module contains de-identified free-text clinical notes for patients.
### Data Loading
<pre>
Database: PostgreSql
</pre>
#### Data Format
The data is available to download as csv files for multiple relational databases. For the purpose of this study PostgreSql is the chosen relational database engine. \
Once the files are downloaded, I followed the steps described [here](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/buildmimic/postgres/README.md) to load data into postgreSql instance.
#### Data Exploration
<pre>
Database Schemas: mimiciv_derived, mimiciv_ed, mimiciv_hosp, mimiciv_icu, mimiciv_note
</pre>
<pre>
Schemas related to study:<br/>mimiciv_hosp,<br/>mimiciv_note
</pre>
MIMIC-IV data is distributed across 5 different schemas. All the data needed for this study is confined in "hosp" and "note" schemas.
<pre>
Tables related to study:<br/>mimiciv_hosp.d_icd_diagnoses,<br/>mimiciv_hosp.icd_diagnoses,<br/>mimiciv_note.discharge
</pre>
**mimiciv_hosp.d_icd_diagnoses** It is a dictionary table and contains all icd codes and their description. The dictionary contains both ICD9 and ICD10 codes. ICD9 is the older version and has been replaced by ICD10 almost a decade ago. Most EHR systems still has ICD9 to cater for historical data. For the purpose of this study we are only focus on ICD10. \
**mimiciv_note.discharge** This table contains the discharge notes of patients for specific hospitalization. \
**mimiciv_hosp.icd_diagnoses** This table contain all the diagnosed icd codes for patient tied to their visit / hospitalization.

## Data Preprocessing
ICD Code Dictionary Stats:
|   |   |
|---|---|
| Total ICD Codes | 109,775 |
| Total ICD 10 Codes | 95,109 |
| Total ICD 10 Codes (Diabetes) | 642 |

### Study Constraints
There are a total of ~110K ICD Codes (including ICD9), whereas ~95K of the total are ICD10 codes. \
The ultimate goal is to predict any of the ICD10 code based on clinical documents but for this study we are going to limit prediction to a subset of codes and are going to limit the data to predict codes only related to Diabetes. \
There are a total of 642 different ICD10 codes in the dictionary related to diabetes which can be categorized into 13 different categories: E08, E09, E10, E11, E13, E23, N25, O24, P70, R73, Z13, Z83, Z86 \
Example: \
E08: Diabetes mellitus due to underlying condition \
E080: Diabetes mellitus due to underlying condition with hyperosmolarity \
E0801: Diabetes mellitus due to underlying condition with hyperosmolarity with coma \
and so on...

### New Schema
Created a new schema to hold subset of data to be used for this study.

- Query to get icd codes specific to diabetes.
```
select icd_code, icd_version, long_title 
into capstone.d_icd_diagnoses
from mimiciv_hosp.d_icd_diagnoses 
where icd_version=10
and long_title ilike '%diabetes%'
```

- Query to get patients and admissions with one or more diabetes diagnosis 
```
select subject_id, hadm_id, seq_num, icd_code, icd_version 
into capstone.diagnoses_icd
from mimiciv_hosp.diagnoses_icd 
where icd_version=10 
and icd_code in (select icd_code from capstone.d_icd_diagnoses)
order by subject_id, hadm_id, seq_num
```
- Query to get discharge notes for patients and admissions tied to a single diabetes diagnosis.
```
with diagnosis as (
	select subject_id, hadm_id from capstone.diagnoses_icd group by subject_id, hadm_id having count(*) = 1
)
select D.subject_id, D.hadm_id, D.note_type, D.note_seq, D.text, null as icd_code
into capstone.discharge_note
from mimiciv_note.discharge D
inner join diagnosis C on D.subject_id=C.subject_id and D.hadm_id=C.hadm_id 
order by D.subject_id, D.hadm_id;
```

### DataFrame for Model Evaluation
Stats on notes related to diabetes diagnosis:
|   |   |
|---|---|
| Total notes count | 25,376 |
| Minimum note length | 697 chars |
| Maximum note length | 53,456 chars |

After filtering data for diabetes diagnosis, we are left with ~25K discharge notes. The smallest note was around 700 characters and the largest note was over 53K characters.

#### DataCleaner utility
<pre>
Language: C#, .NetCore
</pre>

Each note is a free text note and the quality of data varies from one note to other. Each note contains patient demographics, patient medications reconciliation, past medical history, family history, diagnosis, labs and other related details.
A discharge note sample is available [here](data/note_sample.txt). \
The DataCleaner utility is written to extract relevant text for this study. It parses the text and returns lines related to patient history and diagnoses. \
These lines are stored by the utility in the database and converted to csv format to be used as data source for our data frame and modeling. 

#### Final Dataset
<pre>
icd_diagnoses_codes.csv
discharge_note_lines.csv
</pre>
Structure of icd_diagnoses_codes.csv

| Column | Description |
| ------ | ----------- |
| icd_code | The icd code |
| icd_version | The icd version, always 10 | 
| long_title | The description of the code |

Structure of discharge_note_lines.csv

| Column | Description |
| ------ | ----------- |
| line_id| The line id |
| text | The line of text from the discharge note | 
| icd_code | The possible icd code to be predicted |
| valid | True value indicates that this entry has been manually inspected and is correct |

## Techniques / Evaluation
Following technology stack is leveraged for this project:
<pre>
Languages: Python, C#
Python Packages: Pandas, nltk, sklearn, matplotlib, seaborn
Notebook: [lr-comp-analysis](lr-comp-analysis.ipynb)
</pre>

This is a Multi Label Classification problem. I explored following Machine Learning techniques / models:
- Logistic Regression, 
- Naive Bayes,
- State Vector Machines (SVM) 
- AdaBoost Ensemble \

To compare different techniques I use the following metrics: Accuracy, and F-score (Macro and Weighted).

### Text Vectorization
To convert text data to vectors, I leveraged following vectorizers from scikit-learn library. 
- CountVectorizer and 
- TF-IDF

### 1. Logistic Regression

| Vectorizer / Multi Class | Fit Time | Accuracy | F1-Score (macro) | F1-Score (weighted) |
| --------- | -------- | -------- | ---------------- | ------------------- |
| Count Vectorizer / One-vs-Rest | 3.92s | 0.8395 |	0.7056 | 0.7965 |
| Count Vectorizer / Multinomial | 3.08s |	0.8395 | 0.7056 | 0.7965 |
| Count Vectorizer / One-vs-One | 17.11s | 0.8519 | 0.7395 | 0.8225 |
| TFIDF Vectorizer / One-vs-Rest | 2.96s | 0.7778 | 0.6357 | 0.7339 |
| TFIDF Vectorizer / Multinomial | 2.25s | 0.7901 | 0.6573 | 0.7501 |
| TFIDF Vectorizer / One-vs-One | 12.85s | 0.6790 | 0.5060 | 0.6267 |

**Best Performance**:
Logistic Regression (OVO) with Count Vectorizer
Logistic Regression (multi) with TFIDF Vectorizer

### 2. Naive Bayes
| Vectorizer | Fit Time | Accuracy | F1-Score (macro) | F1-Score (weighted) |
| --------- | -------- | -------- | ---------------- | ------------------- |
| Count Vectorizer | 0.48s | 0.7778 | 0.6713 | 0.7479 |
| TFIDF Vectorizer | 0.58s | 0.6543 | 0.4809 | 0.6031 |

**Best Performance**: 
Naive Bayes with Count Vectorizer

### 3. Support Vector Machines
| Vectorizer / Kernel | Fit Time | Accuracy | F1-Score (macro) | F1-Score (weighted) |
| --------- | ------- | -------- | ---------------- | ------------------- |
| Count Vectorizer / RBF | 1.15s | 0.1975 | 0.0300 | 0.0652 |
| Count Vectorizer / Poly | 1.24s | 0.5556 | 0.4647 | 0.5066 |
| TFIDF Vectorizer / RBF | 1.28s | 0.1975 | 0.0300 | 0.0652 |
| TFIDF Vectorizer / Poly | 1.39s | 0.7160 | 0.5839 | 0.6786 |

**Best Performance**: 
SVM with TFIDF Vectorizer and 'Poly' kernel

### 4. AdaBoost Ensemble
| Vectorizer / Multi Class | Fit Time | Accuracy | F1-Score (macro) | F1-Score (weighted) |
| --------- | -------- | -------- | ---------------- | ------------------- |
| Count Vectorizer / OVR | 468.94s | 0.8025 | 0.6602 | 0.7642 |
| TFIDF Vectorizer / Multinomial | 402.24s | 0.7654 | 0.6591 | 0.7392 |
| TFIDF Vectorizer / OVR | 264.45s | 0.8025 | 0.6602 | 0.7642 |
| Count Vectorizer / Multinomial | 211.76s | 0.7654 | 0.6560 | 0.7413 |

**Best Performance**:
AdaBoost using Logistic Regression estimator (OVR) with Count Vectorizer

## Conclusion
To predict the code, we explored four different ML techniques:
- Logistic Regression
- Naive Bayes
- Support Vector Machines
- AdaBoost

For comparision we used the following metrics:
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

## Future Work
While machine learning algorithms documented here and other available algorithms can achieve this task of multi-classification with a certain degree of accuracy but Deep Learning models have shown promising outcomes in text classification and are able to achieve accuracy score of beyond 90%. Exploring Convolutional Neural Networks (CNNs) with NLP techniques (Word embedding, Word2Vector) to tackle this problem will be a possible next step. A good CNN model can significantly improved the prediction performance for code prediction system based.

Retrieval-Augmented Generation (RAG) is a natural language processing (NLP) technique that combines generative and retrieval-based AI models to improve the accuracy and reliability of generative AI models. Exploring use of a Large Language Model (LLM) like Llama with RAG [[5]](#5) might offer the best solution.

## References
<a id="1">[1]</a> Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67. \
<a id="2">[2]</a> Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x \
<a id="3">[3]</a> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. \
<a id="4">[4]</a> Masud JHB, Kuo CC, Yeh CY, Yang HC, Lin MC. Applying Deep Learning Model to Predict Diagnosis Code of Medical Records. Diagnostics (Basel). 2023 Jul 6;13(13):2297. doi: 10.3390/diagnostics13132297. PMID: 37443689; PMCID: PMC10340491. \
<a id="5">[5]</a> https://github.com/run-llama/rags
