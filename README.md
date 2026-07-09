# Riesgo de Inundacion por Parroquia - Esmeraldas, Ecuador

**Live demo:** [https://riesgo-inundacion.onrender.com](https://riesgo-inundacion.onrender.com)

Clasificacion supervisada del nivel de riesgo de inundacion (Bajo/Medio/Alto) para las
parroquias de la provincia de Esmeraldas, Ecuador, usando **19 variables**: 6 climaticas/
geograficas originales + 13 del censo INEC 2022 y MAATE. App Flask con mapa Leaflet
desplegable en PythonAnywhere / Render / Railway.

## Estructura del proyecto

```
├── notebook/                                        # Notebook + datasets asociados
│   ├── Proyecto_2P_Riesgo_Inundacion.ipynb          # Notebook principal (19 features)
│   ├── dataset_riesgo_inundacion_enriquecido.csv    # Dataset enriquecido INEC+MAATE (320x46)
│   ├── dataset_riesgo_inundacion_esmeraldas.csv     # Dataset base
│   ├── modelo_riesgo_inundacion_rf_optimizado.pkl   # Modelo RF optimizado
│   ├── modelo_riesgo_inundacion_voting_soft.pkl     # Voting Classifier
│   ├── label_encoder.pkl                            # Label encoder
│   └── scaler.pkl                                   # Scaler
├── extract/                                         # Scripts de extraccion one-time
│   ├── extraer_datos_sngre.py                       # Extraccion SNGRE REST API
│   └── descargar_geojson_parroquias.py              # Descarga GeoJSON con poligonos
├── build/                                           # Scripts de construccion del dataset
│   ├── construir_dataset_integrado.py               # Integracion datos reales + sinteticos
│   ├── integrar_nuevos_csvs.py                      # Agregacion INEC 2022 + MAATE por parroquia
│   └── generar_dataset_esmeraldas.py                # Generacion del dataset base
└── flask_app/                                       # App web (autocontenida para deploy)
    ├── app.py                                       # API Flask (19 features + lookup INEC)
    ├── requirements.txt                             # Dependencias
    ├── Procfile                                     # Render / Railway
    ├── runtime.txt                                  # Version de Python
    ├── wsgi.py                                      # PythonAnywhere
    ├── templates/index.html                         # Mapa Leaflet + selector parroquia
    ├── static/esmeraldas_parroquias_simple.geojson  # Poligonos parroquiales
    └── models/                                      # Modelos + dataset (para deploy)
        ├── modelo_riesgo_inundacion_rf_optimizado.pkl
        ├── label_encoder.pkl
        └── dataset_riesgo_inundacion_enriquecido.csv
```

## Datos

- **Fuente SNGRE**: Secretaria de Gestion de Riesgos Ecuador - REST API
  - Susceptibilidad a inundaciones, eventos historicos, limites parroquiales (64 parroquias)
- **Censo INEC 2022**: 4 tablas agregadas a nivel de parroquia
  - Poblacion (534K registros): edad media, % menores 15, % mayores 65, % hombres
  - Hogar (130K): % agua publica, % alcantarillado, % electricidad, hacinamiento
  - Vivienda (159K): % pared/techo/piso bueno, personas por dormitorio
  - Emigrantes (6.7K): total emigrantes
- **MAATE**: Concesiones de agua (distancia al cuerpo de agua mas cercano)
- **6 variables clasicas**: Precipitacion, altitud, pendiente, distancia a cuerpos de agua,
  densidad poblacional, area urbanizada
- **13 variables INEC+MAATE**: Incorporadas para cumplir el uso de datos reales del censo
- **Variable objetivo**: Riesgo de inundacion (Bajo/Medio/Alto) construido a partir de
  susceptibilidad real SNGRE + eventos historicos

## Modelos

> Resultados con dataset enriquecido (19 features, 320 parroquias)

| Modelo | Accuracy | F1-Score | CV F1 |
|---|---|---|---|
| Random Forest (opt) | 92.71% | 92.68% | 92.81% |
| Voting Soft | 92.71% | 92.68% | 93.21% |
| Logistic Regression | 91.67% | 91.61% | 91.41% |
| SVM (RBF) | 91.67% | 91.60% | 91.98% |
| Decision Tree | 88.54% | 88.56% | 84.40% |

Top-3 variables mas importantes:
1. DISTANCIA_AGUA_KM (32.1%)
2. PRECIPITACION_ANUAL_MM (16.3%)
3. DENSIDAD_POBLACIONAL (8.7%)

## Ejecucion local

```bash
cd flask_app
pip install -r requirements.txt
python app.py
# Abrir http://127.0.0.1:5000
```

## Notebook

```bash
jupyter notebook notebook/Proyecto_2P_Riesgo_Inundacion.ipynb
```

## Despliegue en produccion

### Render

URL: [https://riesgo-inundacion.onrender.com](https://riesgo-inundacion.onrender.com)

## Funcionalidades de la app web

- Mapa interactivo con poligonos de parroquias coloreados por nivel de riesgo
- **Hover** sobre parroquia: muestra nombre, canton, provincia
- **Click** sobre parroquia: zoom + popup con riesgo y probabilidades
- Leyenda y resumen estadistico
- Formulario de prediccion personalizada con selector de parroquia
  (usa datos censales reales si se selecciona una parroquia)
- Busqueda y listado de parroquias
