import os, json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Rutas relativas - funcionan local y en produccion
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

model = joblib.load(os.path.join(MODELS_DIR, "modelo_riesgo_inundacion_rf_optimizado.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

df = pd.read_csv(
    os.path.join(MODELS_DIR, "dataset_riesgo_inundacion_esmeraldas.csv"),
    encoding="utf-8-sig"
)

FEATURE_COLS = [
    "PRECIPITACION_ANUAL_MM", "ALTITUD_MEDIA_M", "PENDIENTE_PCT",
    "DISTANCIA_AGUA_KM", "DENSIDAD_POBLACIONAL", "AREA_URBANIZADA_PCT",
]

parroquia_features = {}
for (name, canton), group in df.groupby(["PARROQUIA", "CANTON"]):
    key = (name.strip().title(), canton.strip().title())
    parroquia_features[key] = [float(group[c].mean()) for c in FEATURE_COLS]

RIESGO_COLORS = {"Alto": "#e74c3c", "Medio": "#f39c12", "Bajo": "#2ecc71"}


def predecir_riesgo(nombre, canton=""):
    key = (nombre.strip().title(), canton.strip().title())
    if key in parroquia_features:
        X = np.array(parroquia_features[key]).reshape(1, -1)
        pred = model.predict(X)[0]
        riesgo = label_encoder.inverse_transform([pred])[0]
        proba = model.predict_proba(X)[0]
        return riesgo, proba
    for (n, c), feat in parroquia_features.items():
        if n == nombre.strip().title():
            X = np.array(feat).reshape(1, -1)
            pred = model.predict(X)[0]
            riesgo = label_encoder.inverse_transform([pred])[0]
            proba = model.predict_proba(X)[0]
            return riesgo, proba
    return "Desconocido", [0, 0, 0]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/parroquias")
def get_parroquias():
    geojson_path = os.path.join(STATIC_DIR, "esmeraldas_parroquias_simple.geojson")
    if not os.path.exists(geojson_path):
        return jsonify({"error": "GeoJSON no encontrado"}), 404
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    for feature in geojson["features"]:
        props = feature["properties"]
        riesgo, proba = predecir_riesgo(
            props.get("PARROQUIA", ""), props.get("CANTON", "")
        )
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
        data["precipitacion"], data["altitud"], data["pendiente"],
        data["distancia_agua"], data["densidad"], data["area_urbana"],
    ]
    X = np.array(features).reshape(1, -1)
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
