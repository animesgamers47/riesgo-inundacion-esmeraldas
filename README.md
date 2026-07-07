# Riesgo de Inundacion por Parroquia - Esmeraldas, Ecuador

Clasificacion supervisada del nivel de riesgo de inundacion (Bajo/Medio/Alto) para las
parroquias de la provincia de Esmeraldas, Ecuador, usando variables climaticas, geograficas
y demograficas. App Flask desplegable en PythonAnywhere / Render / Railway.

## Estructura del proyecto

```
├── Proyecto_2P_Riesgo_Inundacion.ipynb          # Notebook principal (ejecutado)
├── dataset_riesgo_inundacion_esmeraldas.csv      # Dataset integrado (320 registros)
├── pipeline_riesgo_inundacion.py                 # Pipeline completo de ML
├── extraer_datos_sngre.py                        # Extraccion de datos SNGRE REST API
├── construir_dataset_integrado.py                # Integracion datos reales + sinteticos
├── descargar_geojson_parroquias.py               # Descarga GeoJSON con poligonos
├── modelo_riesgo_inundacion_rf_optimizado.pkl    # Modelo exportado (Pipeline)
├── scaler.pkl / label_encoder.pkl                # Componentes auxiliares
└── flask_app/                                    # App web (autocontenida para deploy)
    ├── app.py                                    # API Flask
    ├── requirements.txt                          # Dependencias
    ├── Procfile                                  # Render / Railway
    ├── runtime.txt                               # Version de Python
    ├── wsgi.py                                   # PythonAnywhere
    ├── templates/index.html                      # Mapa Leaflet interactivo
    ├── static/esmeraldas_parroquias_simple.geojson  # Poligonos parroquiales
    └── models/                                   # Modelos + dataset (para deploy)
        ├── modelo_riesgo_inundacion_rf_optimizado.pkl
        ├── scaler.pkl
        ├── label_encoder.pkl
        └── dataset_riesgo_inundacion_esmeraldas.csv
```

## Datos

- **Fuente principal**: SNGRE (Secretaria de Gestion de Riesgos Ecuador) - REST API
  - Susceptibilidad a inundaciones por parroquia (ZONAS_SUSCEPTIBLES_INUNDACIONES)
  - Eventos historicos de inundacion (COE2, 30 eventos)
  - Limites parroquiales oficiales (64 parroquias)
- **Variables predictoras**: Precipitacion, altitud, pendiente, distancia a cuerpos de agua,
  densidad poblacional, area urbanizada
- **Variable objetivo**: Riesgo de inundacion (Bajo/Medio/Alto) construido a partir de
  susceptibilidad real SNGRE + eventos historicos

## Modelos

| Modelo | Accuracy | F1-Score | CV F1 |
|---|---|---|---|
| Random Forest (opt) | 92.71% | 92.68% | 93.27% |
| Voting Soft | 92.71% | 92.68% | 93.21% |
| Logistic Regression | 91.67% | 91.61% | 91.41% |
| SVM (RBF) | 91.67% | 91.60% | 91.98% |
| Decision Tree | 88.54% | 88.56% | 84.40% |

## Ejecucion local

```bash
cd flask_app
pip install -r requirements.txt
python app.py
# Abrir http://127.0.0.1:5000
```

## Notebook

```bash
jupyter notebook Proyecto_2P_Riesgo_Inundacion.ipynb
```

## Despliegue en produccion

### PythonAnywhere (recomendado)

1. Subir la carpeta `flask_app/` completa via Consola o Web tab (Files)
2. En Web tab: crear nueva app web manual (Python 3.12, Flask)
3. WSGI configuration file: apuntar a `wsgi.py`
4. Static files: `/static/` -> `/home/tu-usuario/flask_app/static/`
5. En una consola: `pip install -r requirements.txt`
6. Recargar la app

### Render

1. Crear nuevo Web Service
2. Conectar repo de GitHub / subir carpeta
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `gunicorn app:app`
5. Subir `flask_app/` como raiz del proyecto (o configurar root directory)

### Railway

1. Crear nuevo proyecto -> Deploy from GitHub
2. Railway detecta `Procfile` automaticamente
3. Configurar root directory como `flask_app/`
4. Build pack: Python

## Funcionalidades de la app web

- Mapa interactivo con poligonos de parroquias coloreados por nivel de riesgo
- **Hover** sobre parroquia: muestra nombre, canton, provincia
- **Click** sobre parroquia: zoom + popup con riesgo y probabilidades
- Leyenda y resumen estadistico
- Formulario de prediccion personalizada
- Busqueda y listado de parroquias

## Licencia

Proyecto academico - Universidad de las Fuerzas Armadas ESPE
