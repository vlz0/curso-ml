# 🕵️‍♂️ Modelo de Detección de Fraudes en Transacciones Bancarias

**Autores:**  
González G. Jerónimo  
Vélez D. Daniel  
Villada C. Juan José  

**Semestre:** 2025-II
---

## 📘 Descripción General

Este proyecto implementa y evalúa modelos supervisados de **Machine Learning** para la **detección de fraude en transacciones financieras móviles**, utilizando el dataset **PaySim**, un simulador basado en datos reales de una empresa africana (López-Rojas et al., 2016).

El objetivo principal es analizar la eficacia de modelos como **XGBoost** y **Random Forest** frente al problema del **desbalance extremo de clases**, aplicando técnicas de *resampling* y evaluando métricas específicas para este tipo de escenarios.

El trabajo forma parte de un estudio académico enfocado en **detección de fraude explicable, reproducible y eficiente** dentro del dominio financiero.

---

## 🧠 Pregunta de Investigación

> ¿Cómo pueden los modelos de aprendizaje automático detectar eficazmente transacciones fraudulentas en dinero móvil, considerando el fuerte desbalance de clases y la necesidad de interpretabilidad básica?

---

## 🎯 Objetivos SMART

- **Specific:** Entrenar y evaluar un modelo supervisado (XGBoost / Random Forest) sobre PaySim.  
- **Measurable:** Medir desempeño mediante **PR-AUC** y **F1-score**, antes y después del balanceo.  
- **Achievable:** Aplicar técnicas de *resampling* (SMOTE) y analizar la importancia de variables.  
- **Relevant:** Evaluar qué factores explican mejor la predicción de fraude.  
- **Time-bound:** Desarrollar y documentar el proyecto durante el semestre académico **2025-II**.

---

## 📊 Dataset: PaySim

- **Fuente:** Kaggle — [PaySim Synthetic Financial Transactions](https://www.kaggle.com/datasets/ealaxi/paysim1)  
- **Tamaño:** ~6.3 millones de registros  
- **Clases:**  
  - `isFraud = 1` → transacciones fraudulentas (~0.1%)  
  - `isFraud = 0` → transacciones legítimas  
- **Variables principales:**
  - `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`
- **Features derivadas:**  
  - `diffOrg = oldbalanceOrg - newbalanceOrig`  
  - `diffDest = newbalanceDest - oldbalanceDest`  
  - `ratio = amount / (oldbalanceOrg + 1)`

---

## ⚙️ Metodología

### 1. Preprocesamiento
- Eliminación de variables con fuga de información (`isFlaggedFraud`).  
- Codificación categórica de `type` mediante *one-hot encoding*.  
- Creación de variables derivadas (`diffOrg`, `diffDest`, `ratio`).  
- Escalado con **MinMaxScaler**.  

### 2. Balanceo de Clases
- Aplicación de **SMOTE (Synthetic Minority Oversampling Technique)** con ratio 1:2 (fraude:no fraude).  
- División del dataset en entrenamiento (85%) y prueba (15%) estratificados.

### 3. Modelado
- **Modelo principal:** XGBoost  
- **Baseline:** Random Forest (15% del dataset balanceado)  
- **Validación cruzada:** 5-fold estratificada  
- **Semilla:** 42 (para reproducibilidad)

### 4. Métricas de Evaluación
- **F1-score**
- **Precision**
- **Recall**
- **PR-AUC (Precision-Recall Area Under Curve)** — métrica principal

---

## 🧾 Resultados Principales

| Modelo | F1-score | Precision | Recall | PR-AUC |
|--------|-----------|-----------|--------|--------|
| **XGBoost** | 0.941 | 0.8911 | 0.9968 | 0.9978 |
| **Random Forest** | 0.9812 | 0.9654 | 0.9976 | — |

📌 El modelo XGBoost detectó prácticamente todos los fraudes, con un F1-score de 0.94 y un PR-AUC de 0.9978.  
Las variables más influyentes fueron `diffOrg`, `newbalanceOrig` y `ratio`.

---

## 💬 Discusión

- **Rendimiento:** el modelo logra alta sensibilidad (recall ≈ 0.99) y buena precisión (~0.89), equilibrando detección y control de falsos positivos.  
- **Interpretabilidad:** las variables derivadas basadas en balances (`diffOrg`) resultaron críticas.  
- **Limitaciones:** uso de datos sintéticos, falta de validación temporal y posibles efectos del oversampling artificial.  
- **Aplicabilidad:** el modelo es viable para sistemas antifraude en tiempo real con monitoreo periódico y revisión manual en casos de incertidumbre.

---

## 🚀 Reproducción del Experimento

### Ejecución en Google Colab
1. Clonar este repositorio o subir los archivos `.ipynb` y `paysim.csv`.  
2. Subir el dataset a la sesión de Colab (`Archivos → Subir`).  
3. Instalar dependencias necesarias:
   ```bash
   !pip install xgboost imbalanced-learn scikit-learn shap
