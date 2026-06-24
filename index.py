import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import kagglehub
import joblib
import os
import streamlit as st

# Download latest version
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")

print("Path to dataset files:", path)
print(os.listdir(path))
df = pd.read_csv(f"{path}/heart.csv")
df.head()
df = df.drop_duplicates()
df.shape
df.columns
df.info()
df.describe()
# checking the missing values
df.isnull().sum()

df.fillna(df.mean(), inplace=True)

# separate the features
X = df.drop("target", axis=1)

y = df["target"]

X_train, X_Test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_Test)
print(X.columns)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_Test)
print(accuracy_score(y_test, rf_pred))

# df["target"].value_counts()
# df.duplicated().sum()
# print(model.get_depth())
# print(model.get_n_leaves())

print(classification_report(y_test, rf_pred))

importance = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})

importance = importance.sort_values(by="Importance", ascending=False, inplace=True)
print(importance)

cm = confusion_matrix(y_test, rf_pred)
print(cm)

scores = cross_val_predict(rf, X, y, cv=5)

print(scores)
print(scores.mean())

importance.head(10).plot(x="Feature", y="Importance", kind="bar")
plt.show

joblib.dump(rf, "heart_disease_model.pkl")
