# Diabetes Prediction using Gradient Boosting Model

## Project Overview
This project focuses on predicting whether an individual is likely to have diabetes based on several health-related features. The dataset used for this project is sourced from Kaggle and contains the following columns:

- **Gender**
- **Age**
- **Hypertension**
- **Heart Disease**
- **Smoking History**
- **BMI (Body Mass Index)**
- **HbA1c Level**
- **Blood Glucose Level**
- **Diabetes (Target)**

The goal of this project is to apply machine learning techniques to predict whether a patient is diabetic or not based on these features. We perform data preprocessing, handle class imbalance with SMOTE (Synthetic Minority Over-sampling Technique), evaluate multiple machine learning models, and then deploy the best-performing model in an easy-to-use web application.

## Dataset Overview
The dataset contains information about the health conditions of patients, including:
- **Gender**: The gender of the patient (Male/Female).
- **Age**: The age of the patient.
- **Hypertension**: Whether the patient has hypertension (1 for Yes, 0 for No).
- **Heart Disease**: Whether the patient has heart disease (1 for Yes, 0 for No).
- **Smoking History**: The patient's smoking history.
- **BMI**: The Body Mass Index of the patient.
- **HbA1c Level**: The Hemoglobin A1c level of the patient.
- **Blood Glucose Level**: The blood glucose level of the patient.
- **Diabetes**: The target variable, indicating whether the patient is diabetic (1 for Yes, 0 for No).

## Data Preprocessing and Transformation

1. **Handling Missing Values**:  
   We first checked the dataset for any missing or null values. If found, they were handled by appropriate imputation or removal.

2. **Feature Engineering**:  
   We didn’t create new features, but we performed essential transformations to the existing ones. All categorical variables were encoded appropriately, and numerical features were scaled.

3. **Handling Class Imbalance**:  
   The target variable "Diabetes" had more non-diabetic patients than diabetic ones. To address this class imbalance, we applied **SMOTE (Synthetic Minority Over-sampling Technique)** to oversample the minority class (diabetic patients) to balance the dataset.

4. **Feature Scaling**:  
   We scaled numerical features like **Age**, **BMI**, **HbA1c Level**, and **Blood Glucose Level** using **StandardScaler** to bring all features onto the same scale, which helps improve model performance.

## Machine Learning Models
We compared the performance of six different machine learning models to predict diabetes:

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Bernoulli Naive Bayes**
4. **Decision Tree**
5. **Random Forest**
6. **Gradient Boosting**

## Model Evaluation
We evaluated the models using the following metrics:
- **Accuracy**: The overall correctness of predictions.
- **Precision**: The fraction of true positive predictions among all positive predictions.
- **Recall**: The fraction of true positives among all actual positives.
- **F1-Score**: The harmonic mean of precision and recall.
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve to measure model separability.

### Model Comparison
The comparison of models based on evaluation metrics is shown in the table below:

| Model                | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression   | 0.89     | 0.41      | 0.88   | 0.56     | 0.96    |
| Random Forest         | 0.96     | 0.74      | 0.76   | 0.75     | 0.97    |
| Decision Tree         | 0.95     | 0.67      | 0.75   | 0.71     | 0.86    |
| Gradient Boosting     | 0.96     | 0.72      | 0.78   | 0.75     | 0.98    |
| KNN                   | 0.91     | 0.48      | 0.81   | 0.60     | 0.92    |
| Bernoulli Naive Bayes | 0.81     | 0.28      | 0.84   | 0.42     | 0.91    |

### Why Gradient Boosting?

While both **Random Forest** and **Gradient Boosting** exhibited exceptional performance, **Gradient Boosting** was selected as the final model due to the following reasons:

1. **Higher AUC-ROC Score**:  
   Gradient Boosting achieved an AUC-ROC score of **0.98**, which is higher than Random Forest's **0.97**. A higher AUC-ROC score indicates better model separability, meaning the model can more accurately differentiate between diabetic and non-diabetic patients.

2. **Better Recall**:  
   Gradient Boosting had a **Recall** of **0.78**, which was higher than Random Forest's **0.76**. This is crucial in medical prediction tasks, as a higher recall ensures that more diabetic patients are correctly identified, reducing the chances of missing out on those in need of medical attention.

3. **Balanced Performance**:  
   Gradient Boosting provided a good balance between **Precision**, **Recall**, and **F1-Score**, making it a well-rounded model for this task. While Random Forest had higher **Precision**, the overall performance of Gradient Boosting, especially with respect to **Recall**, was slightly better, making it the preferred choice for the final model.

## Diabetes Prediction Application

A **web application** was built to allow users to input patient data and predict whether the person is diabetic or not based on the features mentioned earlier. 

### Application Features:
- **Input Fields**: The user can input the following values:
  - Gender
  - Age
  - Hypertension status
  - Heart disease status
  - Smoking history
  - BMI
  - HbA1c level
  - Blood glucose level
- **Output**: The application predicts whether the patient is diabetic (Yes or No) based on the input data.

### Application Workflow:
1. The user inputs the relevant data into the form.
2. The model processes the input data, performs necessary preprocessing, and predicts whether the patient is diabetic.
3. The result is displayed on the screen along with the predicted diabetes status (Yes/No).

## Conclusion
The project demonstrated the use of machine learning algorithms for diabetes prediction, with **Gradient Boosting** being the best model due to its superior **AUC-ROC** and **Recall**. The final model was integrated into a simple web application to make the prediction process accessible and user-friendly.

## Dependencies
To run the project, the following libraries are required:
- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `flask` (for the web application)
- `matplotlib` (for visualizations)
