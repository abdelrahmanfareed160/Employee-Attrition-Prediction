import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


class MLModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.columns = []

    def train_all(self, df):
        X = df.drop("Attrition", axis=1)
        y = df["Attrition"]
        self.columns = X.columns.tolist()

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(),
            "AdaBoost": AdaBoostClassifier(),
        }

        scores = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            scores[name] = score
            self.models[name] = model

        return scores

    def predict(self, df_new):
        df_new = df_new[self.columns]  # Ensure same features
        df_new_scaled = self.scaler.transform(df_new)
        preds = self.models["Random Forest"].predict(df_new_scaled)  # Example
        return preds
