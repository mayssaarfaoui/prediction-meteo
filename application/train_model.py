import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Charger le dataset
data = pd.read_csv("dataset/seattle-weather.csv")

# Nettoyage
data = data.dropna()
data = data.drop("date", axis=1)
le = LabelEncoder()
data["weather"] = le.fit_transform(data["weather"])

# Prétraitement
X = data.drop("weather", axis=1)
y = data["weather"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Entraînement
model = XGBClassifier()
model.fit(X_train, y_train)

# Sauvegarde du modèle
joblib.dump(model, "model_xgboost.pkl")
print("✅ Modèle sauvegardé dans model_xgboost.pkl")
