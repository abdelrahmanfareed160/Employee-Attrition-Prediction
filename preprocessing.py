import pandas as pd
import numpy as np
from sklearn import preprocessing


def preprocess_data(df):
    # Handle missing values (simplified version)
    df = df.dropna(axis=0, thresh=df.shape[1] - 7)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode categorical
    le = preprocessing.LabelEncoder()
    for col in df.select_dtypes(include="object"):
        df[col] = le.fit_transform(df[col].astype(str))

    # Feature engineering (simplified example)
    df['Annual_Income_Per_Year_Tenure'] = (df['Monthly Income'] * 12) / (df['Years at Company'] + 1)

    # Drop ID column
    if "Employee ID" in df.columns:
        df = df.drop("Employee ID", axis=1)

    return df
