"""
Descargar GeoJSON con poligonos de las parroquias de Esmeraldas
desde el servicio REST del SNGRE y guardarlo para la Flask app.
"""
import requests
import json
import os

OUTPUT_DIR = r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico"
URL = "https://sgrportal.gestionderiesgos.gob.ec/server/rest/services/0_GDB_MIN_PUB/LIMITE_PARROQUIAL/MapServer/0/query"

params = {
    "where": "UPPER(dpa_despro) = 'ESMERALDAS'",
    "outFields": "dpa_despro,dpa_descan,dpa_despar,dpa_parroq",
    "returnGeometry": "true",
    "f": "geojson",
    "outSR": "4326",
    "resultRecordCount": 100
}

print("Descargando GeoJSON de parroquias de Esmeraldas...")
resp = requests.get(URL, params=params, timeout=60)
resp.raise_for_status()
data = resp.json()

print(f"Features obtenidas: {len(data.get('features', []))}")

# Guardar GeoJSON raw
geojson_path = os.path.join(OUTPUT_DIR, "flask_app", "static", "esmeraldas_parroquias.geojson")
os.makedirs(os.path.dirname(geojson_path), exist_ok=True)

with open(geojson_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print(f"GeoJSON guardado: {geojson_path}")

# Crear version simplificada (solo properties necesarias + geometry)
simplified = {
    "type": "FeatureCollection",
    "features": []
}

for feat in data.get("features", []):
    props = feat.get("properties", {})
    geom = feat.get("geometry")
    simplified["features"].append({
        "type": "Feature",
        "properties": {
            "DPA_PARROQ": props.get("dpa_parroq", ""),
            "PARROQUIA": props.get("dpa_despar", "").strip(),
            "CANTON": props.get("dpa_descan", "").strip(),
            "PROVINCIA": props.get("dpa_despro", "").strip(),
        },
        "geometry": geom
    })

simple_path = os.path.join(OUTPUT_DIR, "flask_app", "static", "esmeraldas_parroquias_simple.geojson")
with open(simple_path, "w", encoding="utf-8") as f:
    json.dump(simplified, f, ensure_ascii=False)

print(f"GeoJSON simplificado guardado: {simple_path}")
print("Listo!")
