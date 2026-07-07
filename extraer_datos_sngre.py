"""
Extraer datos reales de los servicios REST del SNGRE para Esmeraldas
Usando los nombres de campos correctos descubiertos via exploracion.
"""
import requests
import json
import pandas as pd
import numpy as np
import os

BASE_SNGRE = "https://sgrportal.gestionderiesgos.gob.ec/server/rest/services"
OUTPUT_DIR = r"C:\Users\USER\Desktop\Proyecto Aprendizaje automatico"


def query_geojson(service_url, layer_id, where, out_fields="*",
                  return_geometry=False, max_records=2000):
    """Consulta REST ArcGIS y retorna features como lista de dicts."""
    url = f"{service_url}/{layer_id}/query"
    params = {
        "where": where,
        "outFields": out_fields,
        "returnGeometry": "true" if return_geometry else "false",
        "f": "geojson",
        "resultRecordCount": max_records
    }
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("features", [])
    except Exception as e:
        print(f"  ERROR: {e}")
        return []


# ============================================================
# 1. LIMITES PARROQUIALES DE ESMERALDAS
# ============================================================
print("=" * 60)
print("1. PARROQUIAS DE ESMERALDAS (limites)")
print("=" * 60)

features = query_geojson(
    f"{BASE_SNGRE}/0_GDB_MIN_PUB/LIMITE_PARROQUIAL/MapServer", 0,
    where="UPPER(dpa_despro) = 'ESMERALDAS'",
    out_fields="dpa_parroq,dpa_despar,dpa_canton,dpa_descan,dpa_despro"
)

parroquias = []
for f in features:
    p = f.get("properties", {})
    parroquias.append({
        "DPA_PARROQUIA": p.get("dpa_parroq", ""),
        "PARROQUIA": p.get("dpa_despar", "").strip(),
        "DPA_CANTON": p.get("dpa_canton", ""),
        "CANTON": p.get("dpa_descan", "").strip(),
        "PROVINCIA": p.get("dpa_despro", "").strip(),
    })

df_parr = pd.DataFrame(parroquias)
print(f"  Parroquias encontradas: {len(df_parr)}")
print(f"  Cantones: {sorted(df_parr['CANTON'].unique())}")
df_parr.to_csv(os.path.join(OUTPUT_DIR, "sngre_parroquias_esmeraldas.csv"),
               index=False, encoding="utf-8-sig")
print("  Guardado: sngre_parroquias_esmeraldas.csv")

# ============================================================
# 2. SUSCEPTIBILIDAD A INUNDACIONES (ZONAS_SUSCEPTIBLES)
# ============================================================
print("\n" + "=" * 60)
print("2. SUSCEPTIBILIDAD A INUNDACIONES (poligonos por parroquia)")
print("=" * 60)

# ZONAS_SUSCEPTIBLES_INUNDACIONES tiene INUND_SUI (ALTA/MEDIA/BAJA) por parroquia
features_z = query_geojson(
    f"{BASE_SNGRE}/ZONAS_SUSCEPTIBLES_INUNDACIONES/MapServer", 0,
    where="UPPER(DPA_DESPRO) = 'ESMERALDAS'",
    out_fields="DPA_DESPRO,DPA_DESCAN,DPA_DESPAR,INUND_SUI,SUSCEPTIBILIDAD,Shape_Area",
    return_geometry=True,
    max_records=2000
)

suscept_data = []
for f in features_z:
    p = f.get("properties", {})
    suscept_data.append({
        "PROVINCIA": p.get("DPA_DESPRO", "").strip(),
        "CANTON": p.get("DPA_DESCAN", "").strip(),
        "PARROQUIA": p.get("DPA_DESPAR", "").strip(),
        "INUND_SUI": p.get("INUND_SUI", "").strip(),
        "SUSCEPTIBILIDAD": p.get("SUSCEPTIBILIDAD", "").strip(),
        "AREA_HA": round(p.get("Shape_Area", 0) / 10000, 4),
    })

df_suscept = pd.DataFrame(suscept_data)
print(f"  Poligonos de susceptibilidad: {len(df_suscept)}")
if len(df_suscept) > 0:
    print(f"  Niveles de susceptibilidad: {df_suscept['INUND_SUI'].value_counts().to_dict()}")
    print(f"  Parroquias cubiertas: {df_suscept['PARROQUIA'].nunique()}")

    # Agregar por parroquia: area total y proporcion de area con susceptibilidad alta
    if len(df_suscept) > 0 and 'AREA_HA' in df_suscept.columns:
        agg = df_suscept.groupby('PARROQUIA').agg(
            AREA_SUSCEPTIBLE_HA=('AREA_HA', 'sum'),
            AREA_ALTA_HA=('AREA_HA', lambda x: x[df_suscept.loc[x.index, 'INUND_SUI'] == 'ALTA'].sum()),
            AREA_MEDIA_HA=('AREA_HA', lambda x: x[df_suscept.loc[x.index, 'INUND_SUI'] == 'MEDIA'].sum()),
            AREA_BAJA_HA=('AREA_HA', lambda x: x[df_suscept.loc[x.index, 'INUND_SUI'] == 'BAJA'].sum()),
        ).reset_index()
        agg['PCT_AREA_ALTA'] = (agg['AREA_ALTA_HA'] / agg['AREA_SUSCEPTIBLE_HA'] * 100).round(2)
        agg['PCT_AREA_MEDIA'] = (agg['AREA_MEDIA_HA'] / agg['AREA_SUSCEPTIBLE_HA'] * 100).round(2)
        agg = agg.fillna(0)
        print(f"\n  Resumen por parroquia (top 5):")
        print(agg.head(10).to_string(index=False))
        agg.to_csv(os.path.join(OUTPUT_DIR, "sngre_susceptibilidad_agregada_esmeraldas.csv"),
                   index=False, encoding="utf-8-sig")

df_suscept.to_csv(os.path.join(OUTPUT_DIR, "sngre_susceptibilidad_inundacion_esmeraldas.csv"),
                  index=False, encoding="utf-8-sig")
print("\n  Guardado: sngre_susceptibilidad_inundacion_esmeraldas.csv")

# ============================================================
# 3. EVENTOS DE INUNDACION (COE2)
# ============================================================
print("\n" + "=" * 60)
print("3. EVENTOS DE INUNDACION (COE2 - SNGRE)")
print("=" * 60)

features_coe = query_geojson(
    f"{BASE_SNGRE}/COE2/MapServer", 0,
    where="UPPER(Provincia) = 'ESMERALDAS' AND (UPPER(Evento) LIKE '%INUND%' OR UPPER(Causa) LIKE '%LLUV%')",
    out_fields="Provincia,Canton,Parroquia,Evento,Causa,FechaDelEvento,FamiliasAfectadas,PersonasAfectadasDirectamente,PersonasAfectadasIndirectamente,FamiliasDamnificadas,PersonasDamnificadas,PersonasEvacuadas,ViviendasAfectadas,ViviendasDestruidas,NivelDelEvento,EstadoDelEvento",
    return_geometry=False,
    max_records=5000
)

eventos = []
for f in features_coe:
    p = f.get("properties", {})
    eventos.append({
        "PROVINCIA": p.get("Provincia", "").strip(),
        "CANTON": p.get("Canton", "").strip(),
        "PARROQUIA": p.get("Parroquia", "").strip(),
        "EVENTO": p.get("Evento", "").strip(),
        "CAUSA": p.get("Causa", "").strip(),
        "FECHA": p.get("FechaDelEvento", ""),
        "FAMILIAS_AFECTADAS": p.get("FamiliasAfectadas", 0) or 0,
        "PERSONAS_AFECTADAS": p.get("PersonasAfectadasDirectamente", 0) or 0,
        "DAMNIFICADOS": p.get("PersonasDamnificadas", 0) or 0,
        "EVACUADOS": p.get("PersonasEvacuadas", 0) or 0,
        "VIVIENDAS_AFECTADAS": p.get("ViviendasAfectadas", 0) or 0,
        "VIVIENDAS_DESTRUIDAS": p.get("ViviendasDestruidas", 0) or 0,
        "NIVEL": p.get("NivelDelEvento", ""),
        "ESTADO": p.get("EstadoDelEvento", ""),
    })

df_coe = pd.DataFrame(eventos)
print(f"  Eventos relacionados: {len(df_coe)}")
if len(df_coe) > 0:
    print(f"  Tipos de evento: {df_coe['EVENTO'].value_counts().to_dict()}")
    print(f"  Total personas afectadas: {int(df_coe['PERSONAS_AFECTADAS'].sum())}")
    print(f"  Total viviendas afectadas: {int(df_coe['VIVIENDAS_AFECTADAS'].sum())}")

    # Agregar por parroquia
    agg_coe = df_coe.groupby(['CANTON', 'PARROQUIA']).agg(
        TOTAL_EVENTOS=('EVENTO', 'count'),
        TOTAL_INUNDACIONES=('EVENTO', lambda x: (x.str.contains('INUND', case=False)).sum()),
        TOTAL_AFECTADOS=('PERSONAS_AFECTADAS', 'sum'),
        TOTAL_DAMNIFICADOS=('DAMNIFICADOS', 'sum'),
        TOTAL_VIVIENDAS_AFECTADAS=('VIVIENDAS_AFECTADAS', 'sum'),
    ).reset_index()

    print(f"\n  Eventos por parroquia (top 10):")
    print(agg_coe.sort_values('TOTAL_EVENTOS', ascending=False).head(10).to_string(index=False))
    agg_coe.to_csv(os.path.join(OUTPUT_DIR, "sngre_eventos_agregados_esmeraldas.csv"),
                   index=False, encoding="utf-8-sig")

df_coe.to_csv(os.path.join(OUTPUT_DIR, "sngre_eventos_inundacion_esmeraldas.csv"),
              index=False, encoding="utf-8-sig")
print("\n  Guardado: sngre_eventos_inundacion_esmeraldas.csv")


print("\n" + "=" * 60)
print("EXTRACCION COMPLETADA EXITOSAMENTE")
print("=" * 60)
