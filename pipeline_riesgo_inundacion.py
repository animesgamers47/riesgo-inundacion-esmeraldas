"""
Pipeline completo de Machine Learning para clasificacion de riesgo de inundacion
Provincia de Esmeraldas, Ecuador
Modelos: Logistic Regression, Decision Tree, Random Forest, SVM, Voting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Modelos
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import joblib

# ============================================================
# 1. CARGA DE DATOS
# ============================================================
data_path = r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico\dataset_riesgo_inundacion_esmeraldas.csv"
df = pd.read_csv(data_path, encoding="utf-8-sig")

print("=" * 60)
print("PIPELINE DE CLASIFICACION DE RIESGO DE INUNDACION")
print("Provincia: Esmeraldas, Ecuador")
print("=" * 60)
print(f"\nDataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")

# ============================================================
# 2. ANALISIS EXPLORATORIO (EDA)
# ============================================================
print("\n" + "=" * 60)
print("ANALISIS EXPLORATORIO DE DATOS (EDA)")
print("=" * 60)

print("\n--- Primeros registros ---")
print(df.head())

print("\n--- Distribucion de variable objetivo ---")
print(df["RIESGO_INUNDACION"].value_counts())
print(df["RIESGO_INUNDACION"].value_counts(normalize=True) * 100)

print("\n--- Estadisticas descriptivas ---")
print(df.describe())

print("\n--- Valores nulos ---")
print(df.isnull().sum())

# Visualizaciones EDA
plt.style.use("seaborn-v0_8-darkgrid")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

features_box = ["PRECIPITACION_ANUAL_MM", "ALTITUD_MEDIA_M", "PENDIENTE_PCT",
                "DISTANCIA_AGUA_KM", "DENSIDAD_POBLACIONAL", "AREA_URBANIZADA_PCT"]
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

for i, ax in enumerate(axes.flat):
    feature = features_box[i]
    for j, riesgo in enumerate(["Bajo", "Medio", "Alto"]):
        subset = df[df["RIESGO_INUNDACION"] == riesgo][feature]
        ax.hist(subset, alpha=0.6, label=f"{riesgo}", bins=10,
                color=["#2ecc71", "#f39c12", "#e74c3c"][j])
    ax.set_title(f"Distribucion de {feature.replace('_', ' ').title()}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico", "eda_distribuciones.png"), dpi=150)
plt.close()

# Matriz de correlacion
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[features_box + ["RIESGO_NUM"]].corr()
sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", ax=ax)
ax.set_title("Matriz de Correlacion entre Variables", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico", "eda_correlacion.png"), dpi=150)
plt.close()

# ============================================================
# 3. PREPROCESAMIENTO
# ============================================================
print("\n" + "=" * 60)
print("PREPROCESAMIENTO")
print("=" * 60)

# Variables predictoras y objetivo
feature_cols = ["PRECIPITACION_ANUAL_MM", "ALTITUD_MEDIA_M", "PENDIENTE_PCT",
                "DISTANCIA_AGUA_KM", "DENSIDAD_POBLACIONAL", "AREA_URBANIZADA_PCT"]
X = df[feature_cols].values
y = df["RIESGO_INUNDACION"].values

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Clases: {list(le.classes_)} -> {list(le.transform(le.classes_))}")

# Split entrenamiento/prueba (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)
print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# Escalar características
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. ENTRENAMIENTO DE MODELOS
# ============================================================
print("\n" + "=" * 60)
print("ENTRENAMIENTO DE MODELOS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_split=5, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=4,
        random_state=42, n_jobs=-1
    ),
    "SVM (RBF)": SVC(
        C=10, gamma="scale", kernel="rbf",
        probability=True, random_state=42
    ),
}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
for name, model in models.items():
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # AUC-ROC multiclass (One-vs-Rest)
    if y_proba is not None and y_proba.shape[1] == 3:
        auc_roc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    else:
        auc_roc = None

    results.append({
        "Modelo": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc_roc,
        "CV F1 (mean)": cv_scores.mean(),
        "CV F1 (std)": cv_scores.std(),
    })

    print(f"\n--- {name} ---")
    print(f"  Test Accuracy:  {accuracy:.4f}")
    print(f"  Test F1-Score:  {f1:.4f}")
    print(f"  CV F1 (mean):   {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# ============================================================
# 5. VOTING CLASSIFIER (ENSEMBLE)
# ============================================================
print("\n--- Voting Classifier (Hard + Soft) ---")

voting_hard = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ("dt", DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
        ("svm", SVC(C=10, gamma="scale", probability=True, random_state=42)),
    ],
    voting="hard"
)

voting_soft = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
        ("svm", SVC(C=10, gamma="scale", probability=True, random_state=42)),
    ],
    voting="soft"
)

for vname, vclf in [("Voting Hard", voting_hard), ("Voting Soft", voting_soft)]:
    pipeline_v = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", vclf)
    ])
    pipeline_v.fit(X_train, y_train)
    y_pred_v = pipeline_v.predict(X_test)
    cv_scores_v = cross_val_score(pipeline_v, X_train, y_train, cv=cv, scoring="f1_weighted")

    f1_v = f1_score(y_test, y_pred_v, average="weighted")
    acc_v = accuracy_score(y_test, y_pred_v)

    results.append({
        "Modelo": vname,
        "Accuracy": acc_v,
        "Precision": precision_score(y_test, y_pred_v, average="weighted"),
        "Recall": recall_score(y_test, y_pred_v, average="weighted"),
        "F1-Score": f1_v,
        "AUC-ROC": None,
        "CV F1 (mean)": cv_scores_v.mean(),
        "CV F1 (std)": cv_scores_v.std(),
    })

    print(f"  {vname}:")
    print(f"    Test Accuracy: {acc_v:.4f}, F1-Score: {f1_v:.4f}")
    print(f"    CV F1 (mean):  {cv_scores_v.mean():.4f} +/- {cv_scores_v.std():.4f}")

# ============================================================
# 6. TABLA COMPARATIVA
# ============================================================
print("\n" + "=" * 60)
print("RESULTADOS COMPARATIVOS")
print("=" * 60)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values("F1-Score", ascending=False)
print(df_results.to_string(index=False))

# Guardar tabla
df_results.to_csv(
    os.path.join(r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico", "resultados_modelos.csv"),
    index=False, encoding="utf-8-sig"
)

# ============================================================
# 7. OPTIMIZACION DEL MEJOR MODELO (Random Forest)
# ============================================================
print("\n" + "=" * 60)
print("OPTIMIZACION DE HIPERPARAMETROS (Random Forest)")
print("=" * 60)

param_grid = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [5, 8, 12, None],
    "clf__min_samples_split": [2, 4, 6],
    "clf__min_samples_leaf": [1, 2, 4],
}

pipeline_rf = Pipeline([
    ("scaler", RobustScaler()),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

grid_search = GridSearchCV(
    pipeline_rf, param_grid, cv=5,
    scoring="f1_weighted", n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)

print(f"Mejores parametros: {grid_search.best_params_}")
print(f"Mejor CV F1-Score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
best_f1 = f1_score(y_test, y_pred_best, average="weighted")
best_acc = accuracy_score(y_test, y_pred_best)
print(f"Test F1-Score (optimizado): {best_f1:.4f}")
print(f"Test Accuracy (optimizado): {best_acc:.4f}")

# ============================================================
# 8. MATRIZ DE CONFUSION - MEJOR MODELO
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=le.classes_
)
disp.plot(cmap="Blues", ax=ax, values_format="d")
ax.set_title(f"Matriz de Confusion - Random Forest Optimizado\nF1-Score: {best_f1:.4f}, Accuracy: {best_acc:.4f}", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico", "matriz_confusion_optimizada.png"), dpi=150)
plt.close()

# ============================================================
# 9. CURVA DE APRENDIZAJE
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, y_train, cv=5,
    scoring="f1_weighted", n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color="orange")
ax.plot(train_sizes, train_mean, "o-", color="blue", label="Entrenamiento")
ax.plot(train_sizes, test_mean, "o-", color="orange", label="Validacion")
ax.set_xlabel("Tamaño del conjunto de entrenamiento", fontsize=12)
ax.set_ylabel("F1-Score (weighted)", fontsize=12)
ax.set_title("Curva de Aprendizaje - Random Forest Optimizado", fontsize=14)
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico", "curva_aprendizaje.png"), dpi=150)
plt.close()

# ============================================================
# 10. IMPORTANCIA DE CARACTERISTICAS
# ============================================================
rf_best = best_model.named_steps["clf"]
importances = rf_best.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n" + "=" * 60)
print("IMPORTANCIA DE CARACTERISTICAS")
print("=" * 60)
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(importances)), importances[indices], color="#3498db")
ax.set_yticks(range(len(importances)))
ax.set_yticklabels([feature_cols[i] for i in indices])
ax.set_xlabel("Importancia", fontsize=12)
ax.set_title("Importancia de Caracteristicas - Random Forest", fontsize=14)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico", "importancia_caracteristicas.png"), dpi=150)
plt.close()

# ============================================================
# 11. EXPORTAR MODELO CON JOBLIB
# ============================================================
print("\n" + "=" * 60)
print("EXPORTACION DE MODELOS")
print("=" * 60)

output_dir = r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico"

# Exportar modelo optimizado
joblib.dump(best_model, os.path.join(output_dir, "modelo_riesgo_inundacion_rf_optimizado.pkl"))
print("Modelo RF optimizado exportado: modelo_riesgo_inundacion_rf_optimizado.pkl")

# Exportar Voting Soft
pipeline_voting_soft = Pipeline([
    ("scaler", RobustScaler()),
    ("clf", voting_soft)
])
pipeline_voting_soft.fit(X_train, y_train)
joblib.dump(pipeline_voting_soft, os.path.join(output_dir, "modelo_riesgo_inundacion_voting_soft.pkl"))
print("Modelo Voting Soft exportado: modelo_riesgo_inundacion_voting_soft.pkl")

# Exportar scaler y label encoder por separado
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))
print("Scaler y LabelEncoder exportados: scaler.pkl, label_encoder.pkl")

print("\n" + "=" * 60)
print("PIPELINE COMPLETADO EXITOSAMENTE")
print("=" * 60)
