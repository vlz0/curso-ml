# =====================================
# Modelo de Detección de Fraudes - PaySim
# Versión optimizada para Colab (rápida y completa)
# =====================================

# 1. Librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, average_precision_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =====================================
# 2. Cargar dataset
# (usa el archivo que subiste con files.upload())
# =====================================
df = pd.read_csv("PS_20174392719_1491204439457_log.csv", low_memory=False)
df.columns = df.columns.str.strip()

# =====================================
# 3. Verificar y limpiar la variable objetivo
# =====================================
if "isFraud" not in df.columns:
    raise ValueError("⚠️ La columna 'isFraud' no está en el dataset. Verifica el archivo CSV.")

df = df.dropna(subset=["isFraud"])    
df["isFraud"] = df["isFraud"].astype(int)  

# Eliminar variable con fuga de información
if "isFlaggedFraud" in df.columns:
    df = df.drop(columns=["isFlaggedFraud"])

# =====================================
# 4. Feature Engineering
# =====================================
df["diffOrg"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["diffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
df["ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

# =====================================
# 5. Seleccionar variables
# =====================================
features = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "diffOrg", "diffDest", "ratio"]
target = "isFraud"

X = df[features]
y = df[target]

# =====================================
# 6. División de datos estratificada
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)

# =====================================
# 7. Codificación de variable categórica antes de SMOTE
# =====================================
X_train_encoded = pd.get_dummies(X_train, columns=["type"], drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=["type"], drop_first=True)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# =====================================
# 8. Escalado de variables numéricas
# =====================================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# =====================================
# 9. Balanceo con SMOTE (clase minoritaria)
# =====================================
smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

print(f"Dataset original: {len(X_train)} filas | Balanceado: {len(X_res)} filas")

# =====================================
# 10. Modelo principal: XGBoost
# =====================================
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=(len(y_res[y_res == 0]) / len(y_res[y_res == 1])),
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

print("\nEntrenando modelo principal (XGBoost)...")
xgb.fit(X_res, y_res)
print("Entrenamiento XGBoost completado.\n")

# =====================================
# 11. Evaluación del modelo principal
# =====================================
y_pred = xgb.predict(X_test_scaled)
y_pred_proba = xgb.predict_proba(X_test_scaled)[:, 1]

print("==== Resultados XGBoost ====")
print("F1-score:", round(f1_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("PR-AUC:", round(average_precision_score(y_test, y_pred_proba), 4))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))