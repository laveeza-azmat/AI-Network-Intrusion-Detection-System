# AI-Powered Intrusion Detection System

Information Security
Assignment 1
---

## 1. Project Overview

This project presents a Machine Learning based Network Intrusion Detection System (NIDS) developed for the SecureNet Corp. scenario.

Organizations continuously face network-based threats and malicious activities. Traditional security tools such as firewalls are important, but Machine Learning can provide an additional layer of intelligent threat detection.

The objective of this project is to develop a Machine Learning classification model that automatically classifies network traffic as:

- Normal
- Malicious/Attack

The UNSW-NB15 dataset is used for training and testing the model.

---

## 2. Objectives

The main objectives of this project are:

- Select a realistic network intrusion detection dataset.
- Clean and preprocess network traffic data.
- Perform Exploratory Data Analysis (EDA).
- Convert categorical network features into numerical form.
- Normalize numerical features.
- Develop a Machine Learning classification model.
- Test the model on unseen network traffic.
- Evaluate the model using accuracy, precision, recall and a confusion matrix.
- Analyze the model from a security perspective.

---

## 3. Dataset

### UNSW-NB15

The UNSW-NB15 dataset is a network intrusion detection dataset containing normal network traffic and different types of attack traffic.

The dataset provides predefined training and testing sets.

For this project, the following files are used:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

The dataset contains network traffic features and a binary `label`:

- `0` = Normal traffic
- `1` = Attack traffic

### Official Dataset Source

https://research.unsw.edu.au/projects/unsw-nb15-dataset

---

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- VS Code
- GitHub

---

## 5. Machine Learning Algorithm

### Random Forest Classifier

Random Forest is used as the classification algorithm for this project.

Random Forest combines multiple decision trees to make predictions. It is suitable for this project because network traffic contains multiple numerical and categorical features and may have complex relationships between them.

The model is trained to distinguish between normal and malicious network traffic.

---

## 6. Data Preprocessing

The following preprocessing steps are performed:

1. Load the training and testing datasets using Pandas.
2. Inspect the dataset structure and data types.
3. Check for missing values.
4. Check for duplicate records.
5. Analyze the class distribution.
6. Separate input features from the target label.
7. Remove `attack_cat` from the model input because it describes the attack category.
8. Identify numerical and categorical features.
9. Normalize numerical features using `StandardScaler`.
10. Encode categorical features using `OneHotEncoder`.

A Scikit-learn Pipeline is used to ensure that preprocessing and model training are performed consistently.

---

## 7. Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the network traffic dataset.

The project includes visualizations for:

- Normal versus attack traffic
- Attack category distribution
- Confusion matrix

Generated figures are stored in the `results` folder.

---

## 8. Model Training

The Random Forest classifier is trained using the predefined UNSW-NB15 training dataset.

The testing dataset is kept separate and is used only for evaluating the trained model.

The model therefore makes predictions on previously unseen network traffic.

---

## 9. Model Evaluation

The model is evaluated using security-related performance metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Attack Detection Rate

### Results

The final results will be updated after executing the notebook.

## 10. Security Analysis

Recall is particularly important for an intrusion detection system because false negatives represent attacks that were not detected.

A false negative means:

> Actual attack → Predicted as normal

A high number of false negatives could allow malicious activity to pass through the detection system without generating an alert.

False positives are also important because:

> Actual normal traffic → Predicted as attack

A large number of false positives can generate unnecessary alerts and increase the workload of security analysts.

The confusion matrix is therefore analyzed to understand the security effectiveness of the model.

---

## 11. Project Structure

```text
IS-A1
│
├── dataset/
│   ├── UNSW_NB15_training-set.csv
│   ├── UNSW_NB15_testing-set.csv
│   └── UNSW-NB15_features.csv
│
├── results/
│   ├── class_distribution.png
│   ├── attack_categories.png
│   ├── confusion_matrix.png
│   ├── model_metrics.csv
│   └── ids_random_forest_model.pkl
│
├── python.ipynb
├── README.md
└── requirements.txt