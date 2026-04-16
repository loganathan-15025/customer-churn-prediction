import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("../data/churn.csv")
print("Dataset Loaded Successfully ✅")
print("Shape:", df.shape)
print(df.head())


#data cleaning

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("\nMissing values before cleaning:\n", df.isnull().sum())
df.dropna(inplace=True)
df.drop("customerID", axis=1, inplace=True)
print("\nMissing values after cleaning:\n", df.isnull().sum())
print("\nDataset shape after cleaning:", df.shape)

#feature engineering
df = pd.get_dummies(df, drop_first=True)
print("\nDataset after encoding:")
print(df.head())
print("\nNew shape after encoding:", df.shape)

#train test split

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set:", X_train.shape)
print("Testing set:", X_test.shape)