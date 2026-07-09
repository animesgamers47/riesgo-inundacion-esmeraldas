"""Integrar INEC censo + MAATE al dataset de riesgo de inundacion"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

base = r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico"
print("=== INTEGRACION DE NUEVOS CSVs ===")

# 1. Cargar mapping DPA -> (PARROQUIA, CANTON) desde GeoJSON
with open(os.path.join(base, "flask_app", "static", "esmeraldas_parroquias_simple.geojson"), "r", encoding="utf-8") as f:
    geojson = json.load(f)

dpa_to_parish = {}
for feat in geojson["features"]:
    props = feat["properties"]
    dpa = props["DPA_PARROQ"]
    if dpa is not None:
        dpa = str(dpa).zfill(6)
        dpa_to_parish[dpa] = (props["PARROQUIA"].strip().title(), props["CANTON"].strip().title())

print(f"GeoJSON: {len(dpa_to_parish)} DPA codes mapeados")

# 2. Leer dataset actual
ds = pd.read_csv(os.path.join(base, "dataset_riesgo_inundacion_esmeraldas.csv"), encoding="utf-8-sig")
print(f"Dataset actual: {len(ds)} filas, {len(ds.columns)} columnas")

# 3. FUNCION: agregar INEC por parroquia
def agregar_inec(nombre_csv, col_prefix):
    """Agrega datos de INEC CSV a nivel de parroquia"""
    df = pd.read_csv(os.path.join(base, nombre_csv), encoding="utf-8-sig", low_memory=False)
    
    # Crear DPA code desde I01,I02,I03
    df['DPA'] = df['I01'].astype(str).str.zfill(2) + \
                df['I02'].astype(str).str.zfill(2) + \
                df['I03'].astype(str).str.zfill(2)
    
    # Mapear DPA a (PARROQUIA, CANTON)
    df['PARROQUIA'] = df['DPA'].map(lambda x: dpa_to_parish[x][0] if x in dpa_to_parish else None)
    df['CANTON'] = df['DPA'].map(lambda x: dpa_to_parish[x][1] if x in dpa_to_parish else None)
    
    # Filtrar solo parroquias mapeadas
    df = df.dropna(subset=['PARROQUIA', 'CANTON'])
    
    # Crear key compuesta
    df['KEY'] = df['PARROQUIA'] + '|' + df['CANTON']
    
    print(f"\n  {nombre_csv}: {len(df)} filas, {df['KEY'].nunique()} parroquias mapeadas")
    return df

# ============================================================
# POBLACION - features demograficas
# ============================================================
print("\n--- POBLACION ---")
pob = agregar_inec("Esmeraldas_CSV_Poblacion.csv", "POB")

# P02 = Parentesco, P03 = Edad (numeric), P01 = Sexo (1=M, 2=F)
pob['P03'] = pd.to_numeric(pob['P03'], errors='coerce')
pob['Edad'] = pob['P03'].clip(0, 120)

features_pob = pob.groupby('KEY').agg(
    POB_TOTAL=('P01', 'count'),
    POB_HOMBRES=('P01', lambda x: (x == 1).sum()),
    POB_MUJERES=('P01', lambda x: (x == 2).sum()),
    POB_EDAD_MEDIA=('Edad', 'mean'),
    POB_MENORES_15=('Edad', lambda x: (x < 15).sum()),
    POB_MAYORES_65=('Edad', lambda x: (x >= 65).sum()),
).reset_index()

features_pob['POB_PCT_HOMBRES'] = features_pob['POB_HOMBRES'] / features_pob['POB_TOTAL'] * 100
features_pob['POB_PCT_MUJERES'] = features_pob['POB_MUJERES'] / features_pob['POB_TOTAL'] * 100
features_pob['POB_PCT_MENORES_15'] = features_pob['POB_MENORES_15'] / features_pob['POB_TOTAL'] * 100
features_pob['POB_PCT_MAYORES_65'] = features_pob['POB_MAYORES_65'] / features_pob['POB_TOTAL'] * 100

print(f"  Features poblacion: {list(features_pob.columns)}")

# ============================================================
# HOGAR - servicios basicos
# ============================================================
print("\n--- HOGAR ---")
hog = agregar_inec("Esmeraldas_CSV_Hogar.csv", "HOG")

# H06 = Agua (1=red publica, otro=no), H07=Bano (1=exclusivo, 2=compartido)
# H08=Alcantarillado (1=si, 2=no), H10=Electricidad (1=si, 2=no)
for col in ['H06', 'H07', 'H08', 'H10', 'H11']:
    hog[col] = pd.to_numeric(hog[col], errors='coerce')

features_hog = hog.groupby('KEY').agg(
    HOG_TOTAL=('H06', 'count'),
    HOG_AGUA_PUBLICA=('H06', lambda x: (x == 1).sum()),
    HOG_ALCANTARILLADO=('H08', lambda x: (x == 1).sum()),
    HOG_ELECTRICIDAD=('H10', lambda x: (x == 1).sum()),
    HOG_HACINAMIENTO=('H02', lambda x: pd.to_numeric(x, errors='coerce').mean()),
).reset_index()

features_hog['HOG_PCT_AGUA'] = features_hog['HOG_AGUA_PUBLICA'] / features_hog['HOG_TOTAL'] * 100
features_hog['HOG_PCT_ALCANTARILLADO'] = features_hog['HOG_ALCANTARILLADO'] / features_hog['HOG_TOTAL'] * 100
features_hog['HOG_PCT_ELECTRICIDAD'] = features_hog['HOG_ELECTRICIDAD'] / features_hog['HOG_TOTAL'] * 100

print(f"  Features hogar: {list(features_hog.columns)}")

# ============================================================
# VIVIENDA - calidad de vivienda
# ============================================================
print("\n--- VIVIENDA ---")
viv = agregar_inec("Esmeraldas_CSV_Vivienda.csv", "VIV")

# V01=tipo (1=casa, 2=depto, 3=cuarto, 4=mediagua, 5=rancho, 6=otro)
# V02=paredes (1=hormigon, 2=ladrillo, 3=madera), V03=techo, V04=piso
# TOTPER=total personas, TOTDOR=total dormitorios
for col in ['V01', 'V02', 'V03', 'V04']:
    viv[col] = pd.to_numeric(viv[col], errors='coerce')
viv['TOTPER'] = pd.to_numeric(viv['TOTPER'], errors='coerce')
viv['TOTDOR'] = pd.to_numeric(viv['TOTDOR'], errors='coerce')

features_viv = viv.groupby('KEY').agg(
    VIV_TOTAL=('V01', 'count'),
    VIV_CASA=('V01', lambda x: (x == 1).sum()),
    VIV_PARED_BUENA=('V02', lambda x: (x == 1).sum()),
    VIV_TECHO_BUENO=('V03', lambda x: x.isin([1, 2, 3]).sum()),
    VIV_PISO_BUENO=('V04', lambda x: (x == 1).sum()),
    VIV_TOTPER_MEDIA=('TOTPER', 'mean'),
    VIV_TOTDOR_MEDIA=('TOTDOR', 'mean'),
).reset_index()

features_viv['VIV_PCT_CASA'] = features_viv['VIV_CASA'] / features_viv['VIV_TOTAL'] * 100
features_viv['VIV_PCT_PARED_BUENA'] = features_viv['VIV_PARED_BUENA'] / features_viv['VIV_TOTAL'] * 100
features_viv['VIV_PCT_TECHO_BUENO'] = features_viv['VIV_TECHO_BUENO'] / features_viv['VIV_TOTAL'] * 100
features_viv['VIV_PCT_PISO_BUENO'] = features_viv['VIV_PISO_BUENO'] / features_viv['VIV_TOTAL'] * 100
features_viv['VIV_PERS_POR_DORM'] = features_viv['VIV_TOTPER_MEDIA'] / features_viv['VIV_TOTDOR_MEDIA'].replace(0, np.nan)

print(f"  Features vivienda: {list(features_viv.columns)}")

# ============================================================
# EMIGRANTES
# ============================================================
print("\n--- EMIGRANTES ---")
emi = agregar_inec("Esmeraldas_CSV_Emigrantes.csv", "EMI")

features_emi = emi.groupby('KEY').agg(
    EMI_TOTAL=('M00', 'count'),
).reset_index()

print(f"  Features emigrantes: {list(features_emi.columns)}")

# ============================================================
# MAATE - concesiones de agua
# ============================================================
print("\n--- MAATE AGUA ---")
dfm = pd.read_csv(os.path.join(base, "maate_autorizacionesrecursohidrico_2020diciembre.csv"), 
                  encoding="latin1", sep=";", on_bad_lines="skip")

# Solo Esmeraldas
dfm = dfm[dfm["PROVINCIA_FUENTE"].str.upper().str.strip() == "ESMERALDAS"]
dfm["CAUDAL"] = pd.to_numeric(dfm["CAUDAL_AUTORIZADO_LS"].str.replace(",", "."), errors="coerce")

dfm["KEY"] = dfm["PARROQUIA_FUENTE"].str.strip().str.title() + "|" + dfm["CANTON_FUENTE"].str.strip().str.title()

features_maate = dfm.groupby("KEY").agg(
    MAATE_CONCESIONES=("CAUDAL", "count"),
    MAATE_CAUDAL_TOTAL=("CAUDAL", "sum"),
    MAATE_CAUDAL_MEDIO=("CAUDAL", "mean"),
).reset_index()

print(f"  Features maate: {list(features_maate.columns)}")

# ============================================================
# INTEGRACION FINAL
# ============================================================
print("\n=== INTEGRANDO TODO ===")

# Unir todas las features al dataset actual
ds["KEY"] = ds["PARROQUIA"].str.strip().str.title() + "|" + ds["CANTON"].str.strip().str.title()
print(f"  Dataset keys: {ds['KEY'].nunique()}")

# Merge secuencial
for name, feat_df in [
    ("Poblacion", features_pob),
    ("Hogar", features_hog),
    ("Vivienda", features_viv),
    ("Emigrantes", features_emi),
    ("MAATE", features_maate),
]:
    before = ds.shape[1]
    ds = ds.merge(feat_df, on="KEY", how="left")
    nuevas = ds.shape[1] - before
    print(f"  {name}: +{nuevas} columnas")

print(f"Dataset final: {len(ds)} filas, {len(ds.columns)} columnas")

# Feature columns (nuevas)
nuevas_features = [c for c in ds.columns if c not in [
    'KEY', 'PARROQUIA', 'CANTON', 'CODIGO',
    'PRECIPITACION_ANUAL_MM', 'ALTITUD_MEDIA_M', 'PENDIENTE_PCT',
    'DISTANCIA_AGUA_KM', 'DENSIDAD_POBLACIONAL', 'AREA_URBANIZADA_PCT',
    'PCT_AREA_ALTA', 'PCT_AREA_MEDIA', 'TOTAL_INUNDACIONES', 'TOTAL_AFECTADOS',
    'RIESGO_INUNDACION', 'RIESGO_NUM',
]]
print(f"\nNuevas features disponibles ({len(nuevas_features)}):")
print(nuevas_features)

# Guardar dataset enriquecido
output_path = os.path.join(base, "dataset_riesgo_inundacion_enriquecido.csv")
ds.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\nDataset guardado: {output_path}")

# Mostrar primeras filas
print("\nPrimeras 3 filas:")
print(ds[nuevas_features[:10] + ['RIESGO_INUNDACION']].head(3).to_string())
