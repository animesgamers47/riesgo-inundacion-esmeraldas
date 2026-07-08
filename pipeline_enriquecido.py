"""
Pipeline con dataset enriquecido (INEC censo + MAATE)
Compara features originales vs originales + nuevas
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, roc_auc_score)
import joblib

BASE_DIR = r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico"

# Cargar dataset enriquecido
df = pd.read_csv(os.path.join(BASE_DIR, "dataset_riesgo_inundacion_enriquecido.csv"), encoding="utf-8-sig")
print(f"Dataset enriquecido: {df.shape[0]} filas, {df.shape[1]} columnas")

# Llenar NaN en nuevas features (parroquias sin datos INEC)
fill_cols = [c for c in df.columns if c.startswith(("POB_", "HOG_", "VIV_", "EMI_", "MAATE_"))]
for c in fill_cols:
    if c in df.columns:
        df[c] = df[c].fillna(0)

# Features ORIGINALES (6)
FEATURES_ORIG = [
    "PRECIPITACION_ANUAL_MM", "ALTITUD_MEDIA_M", "PENDIENTE_PCT",
    "DISTANCIA_AGUA_KM", "DENSIDAD_POBLACIONAL", "AREA_URBANIZADA_PCT",
]

# Features NUEVAS (seleccion manual - las mas relevantes)
FEATURES_NUEVAS = [
    # Demograficas
    "POB_EDAD_MEDIA", "POB_PCT_MENORES_15", "POB_PCT_MAYORES_65",
    "POB_PCT_HOMBRES",
    # Servicios basicos
    "HOG_PCT_AGUA", "HOG_PCT_ALCANTARILLADO", "HOG_PCT_ELECTRICIDAD",
    "HOG_HACINAMIENTO",
    # Calidad vivienda
    "VIV_PCT_PARED_BUENA", "VIV_PCT_TECHO_BUENO", "VIV_PCT_PISO_BUENO",
    "VIV_PERS_POR_DORM",
    # Emigracion
    "EMI_TOTAL",
]

FEATURES_ALL = FEATURES_ORIG + FEATURES_NUEVAS
print(f"Features originales: {len(FEATURES_ORIG)}")
print(f"Features nuevas: {len(FEATURES_NUEVAS)}")
print(f"Features total: {len(FEATURES_ALL)}")

# Verificar que todas existen
missing = [c for c in FEATURES_ALL if c not in df.columns]
if missing:
    print(f"FALTAN: {missing}")
    FEATURES_ALL = [c for c in FEATURES_ALL if c in df.columns]
    FEATURES_NUEVAS = [c for c in FEATURES_NUEVAS if c in df.columns]

print(f"\nFeatures a probar:\n  Originales: {FEATURES_ORIG}\n  Nuevas: {FEATURES_NUEVAS}")

# Target
le = LabelEncoder()
y = le.fit_transform(df["RIESGO_INUNDACION"].values)
print(f"\nTarget distribution: {dict(zip(le.classes_, np.bincount(y)))}")

# Modelos
models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
    "SVM (RBF)": SVC(C=10, gamma="scale", probability=True, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluar_modelos(X, y, feature_set_name):
    """Evalua todos los modelos con un conjunto de features dado"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\n{'='*60}")
    print(f"CONJUNTO: {feature_set_name}")
    print(f"Features: {X.shape[1]}")
    print(f"{'='*60}")

    rows = []
    for name, model in models.items():
        pipeline = Pipeline([("scaler", RobustScaler()), ("clf", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted") if y_proba is not None and y_proba.shape[1] == 3 else None

        rows.append({
            "Conjunto": feature_set_name, "Modelo": name,
            "Accuracy": acc, "F1-Score": f1,
            "Precision": prec, "Recall": rec,
            "AUC-ROC": auc, "CV F1": cv_scores.mean(),
        })
        print(f"  {name:25s} Acc={acc:.4f} F1={f1:.4f} CV={cv_scores.mean():.4f}")

    return rows

# ============================================================
# 1. Solo features ORIGINALES
# ============================================================
X_orig = df[FEATURES_ORIG].values
rows_orig = evaluar_modelos(X_orig, y, "Originales (6)")

# ============================================================
# 2. Features ORIGINALES + NUEVAS
# ============================================================
X_all = df[FEATURES_ALL].values
rows_all = evaluar_modelos(X_all, y, "Originales+Nuevas (20)")

# ============================================================
# 3. SOLO features NUEVAS
# ============================================================
X_new = df[FEATURES_NUEVAS].values
rows_new = evaluar_modelos(X_new, y, "Solo Nuevas (14)")

# ============================================================
# 4. Feature Selection: RF con todas + top-K
# ============================================================
print(f"\n{'='*60}")
print("FEATURE IMPORTANCE (RF con todas las features)")
print(f"{'='*60}")

rf_temp = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_temp.fit(RobustScaler().fit_transform(X_all), y)
importances = rf_temp.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop-15 features:")
for i in range(min(15, len(FEATURES_ALL))):
    print(f"  {i+1}. {FEATURES_ALL[indices[i]]}: {importances[indices[i]]:.4f}")

# ============================================================
# 5. GridSearchCV con RANDOM FOREST (features combinadas)
# ============================================================
print(f"\n{'='*60}")
print("GRID SEARCH - Random Forest (Originales+Nuevas)")
print(f"{'='*60}")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.3, random_state=42, stratify=y
)

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
    scoring="f1_weighted", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\nMejores parametros: {grid_search.best_params_}")
print(f"Mejor CV F1: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
acc_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best, average="weighted")
print(f"Test Accuracy: {acc_best:.4f}")
print(f"Test F1-Score: {f1_best:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

# ============================================================
# 6. Comparativa final
# ============================================================
print(f"\n{'='*60}")
print("COMPARATIVA FINAL")
print(f"{'='*60}")

df_results = pd.DataFrame(rows_orig + rows_all + rows_new)
pivot = df_results.pivot_table(
    index="Modelo", columns="Conjunto",
    values=["Accuracy", "F1-Score", "CV F1"],
    aggfunc="first"
)
print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

# Guardar resultados
df_results.to_csv(os.path.join(BASE_DIR, "resultados_comparativos.csv"), index=False, encoding="utf-8-sig")

# ============================================================
# 7. Exportar modelo optimizado
# ============================================================
joblib.dump(best_model, os.path.join(BASE_DIR, "modelo_riesgo_inundacion_rf_optimizado.pkl"))
joblib.dump(le, os.path.join(BASE_DIR, "label_encoder.pkl"))
print(f"\nModelo exportado: {os.path.join(BASE_DIR, 'modelo_riesgo_inundacion_rf_optimizado.pkl')}")
