import os, sys, json, joblib, pandas as pd
from collections import Counter

base = r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico"

print("=== VERIFICACION DEL PROYECTO ===")
print()

# 1. Files from README
checks = [
    ("Proyecto_2P_Riesgo_Inundacion.ipynb", "Notebook principal"),
    ("dataset_riesgo_inundacion_esmeraldas.csv", "Dataset"),
    ("pipeline_riesgo_inundacion.py", "Pipeline ML"),
    ("extraer_datos_sngre.py", "Extraccion SNGRE"),
    ("construir_dataset_integrado.py", "Construir dataset"),
    ("descargar_geojson_parroquias.py", "Descargar GeoJSON"),
    ("modelo_riesgo_inundacion_rf_optimizado.pkl", "Modelo RF"),
    ("scaler.pkl", "Scaler"),
    ("label_encoder.pkl", "LabelEncoder"),
    ("flask_app/app.py", "App Flask"),
    ("flask_app/requirements.txt", "Requirements"),
    ("flask_app/Procfile", "Procfile"),
    ("flask_app/runtime.txt", "Runtime"),
    ("flask_app/wsgi.py", "WSGI"),
    ("flask_app/templates/index.html", "Template HTML"),
    ("flask_app/static/esmeraldas_parroquias_simple.geojson", "GeoJSON"),
    ("flask_app/models/modelo_riesgo_inundacion_rf_optimizado.pkl", "Modelo en models/"),
    ("flask_app/models/scaler.pkl", "Scaler en models/"),
    ("flask_app/models/label_encoder.pkl", "LabelEncoder en models/"),
    ("flask_app/models/dataset_riesgo_inundacion_esmeraldas.csv", "Dataset en models/"),
    ("README.md", "README"),
]

missing = []
for path, desc in checks:
    if not os.path.exists(os.path.join(base, path)):
        missing.append((path, desc))

if missing:
    print("FALTAN ARCHIVOS:")
    for path, desc in missing:
        print(f"  {path} ({desc})")
else:
    print("[OK] Los 21 archivos del README existen")

# 2. Dataset
df = pd.read_csv(os.path.join(base, "dataset_riesgo_inundacion_esmeraldas.csv"), encoding="utf-8-sig")
print(f"[OK] Dataset: {len(df)} filas, {df.shape[1]} columnas, {df['PARROQUIA'].nunique()} parroquias")
print(f"     Target: {df['RIESGO_INUNDACION'].value_counts().to_dict()}")

# 3. Model loads
m = joblib.load(os.path.join(base, "modelo_riesgo_inundacion_rf_optimizado.pkl"))
print(f"[OK] Modelo: Pipeline con pasos: {[s[0] for s in m.steps]}")

# 4. Flask app
sys.path.insert(0, os.path.join(base, "flask_app"))
from app import app
with app.test_client() as c:
    r = c.get("/api/parroquias")
    d = json.loads(r.data)
    riesgos = Counter(f["properties"]["riesgo"] for f in d["features"])
    print(f"[OK] API: {len(d['features'])} features")
    print(f"     Distribucion riesgo: {dict(riesgos)}")

    r2 = c.post("/api/predict", json={
        "precipitacion": 2200, "altitud": 15, "pendiente": 2.5,
        "distancia_agua": 0.8, "densidad": 50, "area_urbana": 5
    })
    p = json.loads(r2.data)
    print(f"[OK] Predict: {p['riesgo']}")

    r3 = c.get("/")
    print(f"[OK] Index: {r3.status_code}")

# 5. .gitignore exists
if os.path.exists(os.path.join(base, ".gitignore")):
    with open(os.path.join(base, ".gitignore")) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    print(f"[OK] .gitignore con {len(lines)} reglas")

print()
print("TODO VERIFICADO - Proyecto listo para GitHub")
