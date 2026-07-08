import os, json, csv
import joblib
import numpy as np
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODELS_DIR, "modelo_riesgo_inundacion_rf_optimizado.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

FEATURE_COLS = [
    "PRECIPITACION_ANUAL_MM", "ALTITUD_MEDIA_M", "PENDIENTE_PCT",
    "DISTANCIA_AGUA_KM", "DENSIDAD_POBLACIONAL", "AREA_URBANIZADA_PCT",
]

INEC_COLS = [
    "POB_EDAD_MEDIA", "POB_PCT_MENORES_15", "POB_PCT_MAYORES_65",
    "POB_PCT_HOMBRES", "HOG_PCT_AGUA", "HOG_PCT_ALCANTARILLADO",
    "HOG_PCT_ELECTRICIDAD", "HOG_HACINAMIENTO", "VIV_PCT_PARED_BUENA",
    "VIV_PCT_TECHO_BUENO", "VIV_PCT_PISO_BUENO", "VIV_PERS_POR_DORM", "EMI_TOTAL",
]

ALL_FEATURE_COLS = FEATURE_COLS + INEC_COLS


def load_csv(filepath):
    """Load CSV as list of dicts without pandas dependency."""
    rows = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_float(row, col, default=0.0):
    try:
        v = row.get(col, default)
        return float(v) if v != "" else default
    except (ValueError, TypeError):
        return default


def groupby_mean(rows, group_cols, val_cols):
    """Group by group_cols and compute mean of val_cols."""
    sums = {}
    counts = {}
    for row in rows:
        key = tuple(row[c].strip().title() for c in group_cols)
        if key not in sums:
            sums[key] = [0.0] * len(val_cols)
            counts[key] = 0
        for i, c in enumerate(val_cols):
            sums[key][i] += get_float(row, c)
        counts[key] += 1
    result = {}
    for key in sums:
        result[key] = [s / counts[key] for s in sums[key]]
    return result


def col_mean(rows, col):
    vals = [get_float(r, col) for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


# Cargar dataset enriquecido
rows_enriched = load_csv(os.path.join(MODELS_DIR, "dataset_riesgo_inundacion_enriquecido.csv"))

# Lookup por parroquia+canton
parroquia_features = groupby_mean(rows_enriched, ["PARROQUIA", "CANTON"], ALL_FEATURE_COLS)

# INEC lookup solo por nombre de parroquia
inec_lookup = {}
for key, vals in parroquia_features.items():
    inec_lookup[key[0]] = vals[len(FEATURE_COLS):]

# INEC promedio global
inec_global_mean = [col_mean(rows_enriched, c) for c in INEC_COLS]

# Cargar dataset original para fallback de features originales
rows_orig = load_csv(os.path.join(MODELS_DIR, "dataset_riesgo_inundacion_esmeraldas.csv"))
orig_means_by_parish = groupby_mean(rows_orig, ["PARROQUIA"], FEATURE_COLS)
orig_global_means = [col_mean(rows_orig, c) for c in FEATURE_COLS]

RIESGO_COLORS = {"Alto": "#e74c3c", "Medio": "#f39c12", "Bajo": "#2ecc71"}


def predecir_riesgo(nombre, canton="", inec_vals=None):
    key = (nombre.strip().title(), canton.strip().title())
    if key in parroquia_features:
        X = np.array(parroquia_features[key]).reshape(1, -1)
    else:
        nombre_key = nombre.strip().title()
        orig = orig_means_by_parish.get(
            (nombre_key,),
            np.array([orig_global_means])
        )
        if isinstance(orig, list):
            orig = np.array(orig)
        if inec_vals is None:
            inec_vals = inec_lookup.get(nombre_key, inec_global_mean)
        X = np.concatenate([orig.flatten(), np.array(inec_vals)]).reshape(1, -1)

    pred = model.predict(X)[0]
    riesgo = label_encoder.inverse_transform([pred])[0]
    proba = model.predict_proba(X)[0]
    return riesgo, proba


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/parroquias")
def get_parroquias():
    geojson_path = os.path.join(BASE_DIR, "static", "esmeraldas_parroquias_simple.geojson")
    if not os.path.exists(geojson_path):
        return jsonify({"error": "GeoJSON no encontrado"}), 404

    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    for feature in geojson["features"]:
        props = feature["properties"]
        nombre = props.get("PARROQUIA", "")
        canton = props.get("CANTON", "")
        riesgo, proba = predecir_riesgo(nombre, canton)
        props["riesgo"] = riesgo
        props["color"] = RIESGO_COLORS.get(riesgo, "#95a5a6")
        props["proba_bajo"] = round(float(proba[1]), 3)
        props["proba_medio"] = round(float(proba[2]), 3)
        props["proba_alto"] = round(float(proba[0]), 3)

    return jsonify(geojson)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [
        data.get("precipitacion", 0),
        data.get("altitud", 0),
        data.get("pendiente", 0),
        data.get("distancia_agua", 0),
        data.get("densidad", 0),
        data.get("area_urbana", 0),
    ]
    # Si se especifica parroquia, usar sus datos INEC
    parroquia = data.get("parroquia", "").strip().title()
    if parroquia and parroquia in inec_lookup:
        inec_vals = inec_lookup[parroquia]
    else:
        inec_vals = inec_global_mean

    X = np.array(features + inec_vals).reshape(1, -1)
    pred = model.predict(X)[0]
    riesgo = label_encoder.inverse_transform([pred])[0]
    proba = model.predict_proba(X)[0]

    return jsonify({
        "riesgo": riesgo,
        "color": RIESGO_COLORS.get(riesgo, "#95a5a6"),
        "proba_bajo": round(float(proba[1]), 3),
        "proba_medio": round(float(proba[2]), 3),
        "proba_alto": round(float(proba[0]), 3),
    })


@app.route("/api/parroquias-list")
def parroquias_list():
    return jsonify([
        {"nombre": k[0], "canton": k[1]}
        for k in sorted(parroquia_features.keys())
    ])


if __name__ == "__main__":
    app.run(debug=True)
