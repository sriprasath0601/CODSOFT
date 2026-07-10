import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
df = pd.read_csv(r"C:\Users\Divya\Desktop\Titanic-Dataset.csv")
print("\n========== FIRST 5 ROWS ==========")
print(df.head())
print("\n========== LAST 5 ROWS ==========")
print(df.tail())
print("\n========== RANDOM 10 ROWS ==========")
print(df.sample(10))
print("\n========== DATASET SHAPE ==========")
print(df.shape)
print("\n========== DATASET INFORMATION ==========")
print(df.info())
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop(["Cabin", "PassengerId", "Name", "Ticket"], axis=1, inplace=True)
encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])
plt.figure(figsize=(6,5))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()
plt.figure(figsize=(6,5))
sns.countplot(x="Sex", data=df)
plt.title("Gender Distribution")
plt.show()
plt.figure(figsize=(6,5))
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class Distribution")
plt.show()
plt.figure(figsize=(6,5))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()
plt.figure(figsize=(6,5))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()
plt.figure(figsize=(8,5))
sns.histplot(df["Fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.show()
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
plt.figure(figsize=(8,5))
sns.boxplot(x="Survived", y="Age", data=df)
plt.title("Age vs Survival")
plt.show()
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("\n========== MODEL ACCURACY ==========")
print("Accuracy :", accuracy)
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, predictions))
print("\n========== CONFUSION MATRIX ==========")
cm = confusion_matrix(y_test, predictions)
print(cm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Survived", "Survived"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})
print("\n========== SAMPLE PREDICTIONS ==========")
print(comparison.head(20))
importance = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)
print("\n========== FEATURE IMPORTANCE ==========")
print(importance_df)
plt.figure(figsize=(8,5))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()
scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("\n========== CROSS VALIDATION ==========")
print("Scores :", scores)
print("Average Accuracy :", scores.mean())
sample = [[
    3,      # Pclass
    1,      # Sex (Male=1, Female=0)
    25,     # Age
    0,      # SibSp
    0,      # Parch
    7.25,   # Fare
    2       # Embarked
]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nPrediction : Passenger Survived")
else:
    print("\nPrediction : Passenger Did Not Survive")

joblib.dump(model, "titanic_survival_model.pkl")

print("\nModel saved successfully as titanic_survival_model.pkl")
