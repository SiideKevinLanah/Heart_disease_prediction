Heart Disease Prediction using Machine Learning
Overview

This project predicts the likelihood of heart disease using Machine Learning algorithms. The model is trained on a heart disease dataset and deployed through a Streamlit web application where users can enter patient information and receive a prediction.

Features
Data preprocessing and cleaning
Duplicate record detection and removal
Decision Tree Classification
Random Forest Classification
Model evaluation using accuracy metrics
Feature importance analysis
Interactive Streamlit web application
Saved trained model using Joblib
Dataset

The project uses the Heart Disease Dataset containing medical attributes such as:

Age
Sex
Chest Pain Type (cp)
Resting Blood Pressure (trestbps)
Cholesterol (chol)
Fasting Blood Sugar (fbs)
Resting ECG Results (restecg)
Maximum Heart Rate (thalach)
Exercise Induced Angina (exang)
Oldpeak
Slope
Number of Major Vessels (ca)
Thal
Target (Heart Disease)
Data Cleaning

During exploratory data analysis, 723 duplicate records were identified and removed from the dataset. This helped prevent data leakage and produced more realistic model evaluation results.

Models Used
Decision Tree Classifier
Accuracy: 73.77%
Random Forest Classifier
Accuracy: 85.25%

The Random Forest model outperformed the Decision Tree model and was selected as the final model for deployment.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Joblib
Streamlit
Matplotlib
Installation

Clone the repository:

git clone <repository-url>
cd HeartDiseasePrediction

Install dependencies:

pip install -r requirements.txt
Run the Application
streamlit run app.py
