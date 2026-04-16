import pandas as pd

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