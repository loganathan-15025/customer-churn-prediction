import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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

#random forest model

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nBaseline Random Forest Accuracy:", rf_acc)

#model improvement using gridsearchcv

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 10, 12],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_

print("\nBest Parameters:", rf_grid.best_params_)

rf_pred_tuned = rf_best.predict(X_test)

rf_acc_tuned = accuracy_score(y_test, rf_pred_tuned)

print("Tuned Random Forest Accuracy:", rf_acc_tuned)