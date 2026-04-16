import pandas as pd

df = pd.read_csv("../data/churn.csv")
print("Dataset Loaded Successfully ✅")
print("Shape:", df.shape)
print(df.head())