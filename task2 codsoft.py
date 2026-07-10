import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv(
    r"C:\Users\Divya\Desktop\IMDb Movies India.csv",
    encoding="cp1252"
)
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

print("\n========== RANDOM 10 RECORDS ==========")
print(df.sample(10))

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())
df["Year"] = df["Year"].str.extract("(\d{4})")

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Duration"] = df["Duration"].str.replace(" min", "", regex=False)

df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

df["Votes"] = df["Votes"].str.replace(",", "", regex=False)

df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

df["Year"] = df["Year"].fillna(df["Year"].median())
df["Duration"] = df["Duration"].fillna(df["Duration"].median())
df["Votes"] = df["Votes"].fillna(df["Votes"].median())

df["Genre"] = df["Genre"].fillna(df["Genre"].mode()[0])
df["Director"] = df["Director"].fillna(df["Director"].mode()[0])

df["Actor 1"] = df["Actor 1"].fillna("Unknown")
df["Actor 2"] = df["Actor 2"].fillna("Unknown")
df["Actor 3"] = df["Actor 3"].fillna("Unknown")
# Remove rows where Rating is missing
df.dropna(subset=["Rating"], inplace=True)
plt.figure(figsize=(8,5))
sns.histplot(df["Rating"], bins=20, kde=True)
plt.title("Movie Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(df["Year"], bins=20)
plt.title("Movies Released by Year")
plt.xlabel("Year")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Duration"], bins=20)
plt.title("Movie Duration Distribution")
plt.xlabel("Duration")
plt.show()
plt.figure(figsize=(8,5))
sns.histplot(df["Votes"], bins=20)
plt.title("Votes Distribution")
plt.xlabel("Votes")
plt.show()

plt.figure(figsize=(10,6))
df["Genre"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,6))
df["Director"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Directors")
plt.xlabel("Director")
plt.ylabel("Movies Directed")
plt.xticks(rotation=45)
plt.show()
encoder = LabelEncoder()

for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]:
    df[col] = encoder.fit_transform(df[col])

plt.figure(figsize=(10,8))
sns.heatmap(
    df[["Genre","Director","Actor 1","Actor 2","Actor 3",
        "Year","Duration","Votes","Rating"]].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

X = df[[
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3",
    "Year",
    "Duration",
    "Votes"
]]

y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\n========== MODEL PERFORMANCE ==========")

print("Mean Absolute Error :", mean_absolute_error(y_test, pred))

print("Mean Squared Error :", mean_squared_error(y_test, pred))

print("Root Mean Squared Error :", np.sqrt(mean_squared_error(y_test, pred)))

print("R2 Score :", r2_score(y_test, pred))

comparison = pd.DataFrame({
    "Actual Rating": y_test.values,
    "Predicted Rating": pred
})

print("\n========== SAMPLE PREDICTIONS ==========")
print(comparison.head(15))

plt.figure(figsize=(8,6))
plt.scatter(y_test, pred)

plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Ratings")

plt.show()

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
    cv=5,
    scoring="r2"
)

print("\n========== CROSS VALIDATION ==========")

print("Scores :", scores)

print("Average R2 :", scores.mean())

sample = [[
    df["Genre"].iloc[0],
    df["Director"].iloc[0],
    df["Actor 1"].iloc[0],
    df["Actor 2"].iloc[0],
    df["Actor 3"].iloc[0],
    2023,
    150,
    50000
]]

prediction = model.predict(sample)

print("\nPredicted Movie Rating :", prediction[0])

joblib.dump(model, "movie_rating_prediction.pkl")

print("\nModel saved successfully as movie_rating_prediction.pkl")

plt.figure(figsize=(8,6))
residuals = y_test - pred
plt.scatter(pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Rating")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
comparison = comparison.head(30)

plt.figure(figsize=(12,6))
plt.plot(comparison["Actual Rating"].values, label="Actual", marker="o")
plt.plot(comparison["Predicted Rating"].values, label="Predicted", marker="x")
plt.title("Actual vs Predicted Ratings")
plt.xlabel("Movies")
plt.ylabel("Rating")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.bar(importance_df["Feature"], importance_df["Importance"])
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.show()
