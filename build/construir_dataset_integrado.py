"""
Construir dataset integrado con datos reales SNGRE + predictoras sinteticas
para clasificacion de riesgo de inundacion en parroquias de Esmeraldas.
"""
import pandas as pd
import numpy as np
import os

OUTPUT_DIR = r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico"

# ============================================================
# CARGAR DATOS REALES DEL SNGRE
# ============================================================

# Parroquias oficiales
df_parr = pd.read_csv(os.path.join(OUTPUT_DIR, "sngre_parroquias_esmeraldas.csv"), encoding="utf-8-sig")
print(f"Parroquias oficiales: {len(df_parr)}")

# Susceptibilidad agregada
suscept_path = os.path.join(OUTPUT_DIR, "sngre_susceptibilidad_agregada_esmeraldas.csv")
if os.path.exists(suscept_path):
    df_suscept = pd.read_csv(suscept_path, encoding="utf-8-sig")
    print(f"Susceptibilidad agregada: {len(df_suscept)} parroquias")
else:
    df_suscept = pd.DataFrame()

# Eventos agregados
eventos_path = os.path.join(OUTPUT_DIR, "sngre_eventos_agregados_esmeraldas.csv")
if os.path.exists(eventos_path):
    df_eventos = pd.read_csv(eventos_path, encoding="utf-8-sig")
    print(f"Eventos agregados: {len(df_eventos)} parroquias")
else:
    df_eventos = pd.DataFrame()

# ============================================================
# CONSTRUIR TARGET VARIABLE (RIESGO) basado en datos reales
# ============================================================
# Combinar susceptibilidad + eventos para asignar nivel de riesgo real

# Merge susceptibilidad con parroquias
df_target = df_parr[['PARROQUIA', 'CANTON']].copy()

if len(df_suscept) > 0:
    df_suscept = df_suscept.rename(columns={'PARROQUIA': 'PARROQUIA'})
    # Normalizar nombres
    df_suscept['PARROQUIA'] = df_suscept['PARROQUIA'].str.upper().str.strip()
    df_parr['PARROQUIA_U'] = df_parr['PARROQUIA'].str.upper().str.strip()
    df_target['PARROQUIA_U'] = df_target['PARROQUIA'].str.upper().str.strip()

    df_target = df_target.merge(
        df_suscept[['PARROQUIA', 'PCT_AREA_ALTA', 'PCT_AREA_MEDIA', 'AREA_SUSCEPTIBLE_HA']],
        left_on='PARROQUIA_U', right_on='PARROQUIA', how='left'
    )
    df_target = df_target.drop(columns=['PARROQUIA_y'], errors='ignore')
    df_target = df_target.rename(columns={'PARROQUIA_x': 'PARROQUIA'})

if len(df_eventos) > 0:
    df_eventos['PARROQUIA_U'] = df_eventos['PARROQUIA'].str.upper().str.strip()
    df_target = df_target.merge(
        df_eventos[['PARROQUIA_U', 'TOTAL_EVENTOS', 'TOTAL_INUNDACIONES',
                     'TOTAL_AFECTADOS', 'TOTAL_VIVIENDAS_AFECTADAS']],
        on='PARROQUIA_U', how='left'
    )

# Llenar NaN con 0
for col in ['PCT_AREA_ALTA', 'PCT_AREA_MEDIA', 'AREA_SUSCEPTIBLE_HA',
            'TOTAL_EVENTOS', 'TOTAL_INUNDACIONES', 'TOTAL_AFECTADOS',
            'TOTAL_VIVIENDAS_AFECTADAS']:
    if col in df_target.columns:
        df_target[col] = df_target[col].fillna(0)

# ============================================================
# ASIGNAR RIESGO BASADO EN DATOS REALES
# ============================================================
def asignar_riesgo(row):
    """Asignar riesgo basado en susceptibilidad real y eventos historicos."""
    # Puntaje de susceptibilidad (0-100)
    score = 0
    if 'PCT_AREA_ALTA' in row.index and row['PCT_AREA_ALTA'] > 0:
        score += min(row['PCT_AREA_ALTA'] * 0.5, 40)
    if 'PCT_AREA_MEDIA' in row.index:
        score += min(row['PCT_AREA_MEDIA'] * 0.2, 20)

    # Puntaje eventos historicos
    if 'TOTAL_INUNDACIONES' in row.index:
        score += min(row['TOTAL_INUNDACIONES'] * 8, 30)
    if 'TOTAL_AFECTADOS' in row.index:
        score += min(row['TOTAL_AFECTADOS'] * 0.1, 10)

    if score >= 35:
        return "Alto"
    elif score >= 15:
        return "Medio"
    else:
        return "Bajo"

df_target['RIESGO_INUNDACION'] = df_target.apply(asignar_riesgo, axis=1)
riesgo_map = {"Bajo": 0, "Medio": 1, "Alto": 2}
df_target['RIESGO_NUM'] = df_target['RIESGO_INUNDACION'].map(riesgo_map)

print(f"\nDistribucion de riesgo (real):")
print(df_target['RIESGO_INUNDACION'].value_counts())

# ============================================================
# GENERAR VARIABLES PREDICTORAS PARA CADA PARROQUIA
# ============================================================
# Valores base para cada parroquia basados en ubicacion geografica
# Usar datos realistas asignados por posicion geografica

def generar_predictoras(row, riesgo, seed=42):
    """Generar valores de variables predictoras correlacionadas con el nivel de riesgo.
    
    Para Bajo: mayor altitud, mayor pendiente, lejos de agua, menos denso, menos urbano.
    Para Alto: menor altitud, menor pendiente, cerca de agua, mas denso, mas urbano.
    Para Medio: valores intermedios.
    """
    rng = np.random.RandomState(seed + hash(row['PARROQUIA_U']) % 1000)

    canton = row.get('CANTON', '')

    # Precipitacion base por canton (datos INAMHI reales)
    precip_base = {
        'ESMERALDAS': 1900, 'ATACAMES': 2050, 'ELOY ALFARO': 3000,
        'MUISNE': 2350, 'QUININD': 2450, 'RIOVERDE': 2100, 'SAN LORENZO': 2900
    }
    # Altitud base por canton
    alt_base = {
        'ESMERALDAS': 22, 'ATACAMES': 12, 'ELOY ALFARO': 14,
        'MUISNE': 8, 'QUININD': 42, 'RIOVERDE': 14, 'SAN LORENZO': 14
    }

    bp = precip_base.get(str(canton).upper()[:9], 2200)
    ba = alt_base.get(str(canton).upper()[:9], 20)

    # Parametros segun riesgo
    if riesgo == 'Alto':
        # Cerca de rios, baja pendiente, alta urbanizacion, alta densidad
        precip_mul = 1.05
        pend_m = 1.5
        dist_m = 0.4
        dens_m = 1.5
        urb_m = 1.8
    elif riesgo == 'Medio':
        # Valores intermedios
        precip_mul = 1.0
        pend_m = 2.5
        dist_m = 0.9
        dens_m = 1.0
        urb_m = 1.0
    else:  # Bajo
        # Lejos de agua, alta pendiente, baja densidad
        precip_mul = 0.95
        pend_m = 3.8
        dist_m = 1.6
        dens_m = 0.7
        urb_m = 0.5

    area_alta = row.get('PCT_AREA_ALTA', 0) if not pd.isna(row.get('PCT_AREA_ALTA', 0)) else 0

    precip = round(bp * precip_mul + rng.normal(0, 60), 1)
    altitud = round(max(1, ba + rng.normal(0, 4)), 1)
    pendiente = round(max(0.3, pend_m + rng.normal(0, 0.5)), 2)
    dist_agua = round(max(0.05, dist_m + rng.normal(0, 0.15)), 2)

    es_urbana = 'CABECERA' in str(row.get('PARROQUIA', ''))
    if es_urbana:
        densidad = round(300 * dens_m + rng.normal(0, 80), 1)
        urb_pct = round(30 * urb_m + rng.normal(0, 8), 1)
    else:
        # Misma logica pero escala menor
        densidad = round(50 * dens_m + rng.normal(0, 20), 1)
        urb_pct = round(5 * urb_m + rng.normal(0, 3), 1)

    return {
        'PRECIPITACION_ANUAL_MM': max(800, precip),
        'ALTITUD_MEDIA_M': altitud,
        'PENDIENTE_PCT': pendiente,
        'DISTANCIA_AGUA_KM': dist_agua,
        'DENSIDAD_POBLACIONAL': max(5, densidad),
        'AREA_URBANIZADA_PCT': max(0, min(100, urb_pct)),
    }

# Generar dataset final con variaciones (5 variaciones por parroquia)
rows = []
for _, row in df_target.iterrows():
    riesgo = row['RIESGO_INUNDACION']
    risco_num = row['RIESGO_NUM']

    for variacion in range(5):
        pred = generar_predictoras(row, riesgo, seed=42 + variacion * 10)
        # Pequeno ruido en el target para casos borderline
        r_num = risco_num
        if np.random.random() < 0.08:
            r_num = max(0, min(2, risco_num + np.random.choice([-1, 1])))

        rows.append({
            'PARROQUIA': row['PARROQUIA'],
            'CANTON': row['CANTON'],
            'PROVINCIA': 'ESMERALDAS',
            **pred,
            'RIESGO_INUNDACION': riesgo if r_num == risco_num else list(riesgo_map.keys())[list(riesgo_map.values()).index(r_num)],
            'RIESGO_NUM': r_num,
        })

df_final = pd.DataFrame(rows)
print(f"\nDataset integrado: {len(df_final)} registros, {df_final.shape[1]} columnas")
print(f"Distribucion de riesgo:")
print(df_final['RIESGO_INUNDACION'].value_counts())
print(f"\nEstadisticas:")
print(df_final.describe())

# Guardar
df_final.to_csv(os.path.join(OUTPUT_DIR, "dataset_riesgo_inundacion_esmeraldas.csv"),
                index=False, encoding="utf-8-sig")
print(f"\nGuardado: dataset_riesgo_inundacion_esmeraldas.csv")
