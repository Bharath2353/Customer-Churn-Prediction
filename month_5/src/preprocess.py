import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(df[col])

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    columns = X.columns.tolist()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, columns, scaler