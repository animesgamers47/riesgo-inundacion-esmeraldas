"""Generar informe PDF del proyecto (formato ensayo academico)"""
from fpdf import FPDF
import os

FONT_DIR = r'C:\Windows\Fonts'

class Informe(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font('TNR', '', os.path.join(FONT_DIR, 'times.ttf'))
        self.add_font('TNR', 'B', os.path.join(FONT_DIR, 'timesbd.ttf'))
        self.add_font('TNR', 'I', os.path.join(FONT_DIR, 'timesi.ttf'))
        self.add_font('TNR', 'BI', os.path.join(FONT_DIR, 'timesbi.ttf'))

    def header(self):
        if self.page_no() > 1:
            self.set_font('TNR', 'I', 9)
            self.cell(0, 8, 'Aprendizaje Automatico | Proyecto del Segundo Parcial', align='C')
            self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font('TNR', 'I', 9)
        self.cell(0, 10, f'{self.page_no()}', align='C')

    def titulo(self, txt):
        self.set_font('TNR', 'B', 13)
        self.multi_cell(0, 7, txt)
        self.ln(3)

    def subtitulo(self, txt):
        self.set_font('TNR', 'B', 11)
        self.multi_cell(0, 6, txt)
        self.ln(2)

    def parrafo(self, txt):
        self.set_font('TNR', '', 11)
        self.multi_cell(0, 5.5, txt)
        self.ln(1)

    def insertar_tabla(self, headers, data, col_widths):
        self.set_font('TNR', 'B', 10)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, align='C')
        self.ln()
        self.set_font('TNR', '', 10)
        for row in data:
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, str(val), border=1, align='C')
            self.ln()
        self.ln(3)


pdf = Informe()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# ========== PORTADA ==========
pdf.ln(25)
pdf.set_font('TNR', 'B', 20)
pdf.multi_cell(0, 10, 'Clasificacion del Riesgo de Inundacion\npor Parroquia en la Provincia de Esmeraldas,\nEcuador', align='C')
pdf.ln(8)
pdf.set_font('TNR', '', 14)
pdf.cell(0, 8, 'Proyecto del Segundo Parcial - Aprendizaje Automatico', align='C')
pdf.ln(8)
pdf.cell(0, 8, 'Julio 2026', align='C')
pdf.ln(6)
pdf.set_font('TNR', 'I', 11)
pdf.cell(0, 7, 'Grupo B', align='C')
pdf.ln(7)
pdf.cell(0, 7, 'Universidad de Guayaquil', align='C')
pdf.ln(7)
pdf.cell(0, 7, 'Facultad de Ciencias Matematicas y Fisicas', align='C')

# ========== 1. INTRODUCCION ==========
pdf.add_page()
pdf.titulo('1. Introduccion y Contextualizacion')
pdf.parrafo(
    'Ecuador es uno de los paises de America Latina con mayor vulnerabilidad frente a eventos '
    'hidrometeorologicos extremos. Su ubicacion geografica, la presencia de la cordillera de los Andes '
    'y la influencia de corrientes oceanicas como El Nino generan condiciones que favorecen lluvias '
    'torrenciales e inundaciones recurrentes. La provincia de Esmeraldas, ubicada en la region Costa '
    'del Ecuador, es particularmente afectada por estos fenomenos debido a su alta pluviosidad '
    '(que supera los 2,000 mm anuales en muchas parroquias), su topografia de baja altitud '
    '(altitud media de 16.9 m) y la presencia de numerosos rios y esteros que atraviesan zonas '
    'densamente pobladas.'
)
pdf.parrafo(
    'Segun datos de la Secretaria de Gestion de Riesgos (SNGRE), en la ultima decada se han '
    'registrado mas de 30 eventos de inundacion en la provincia, afectando a miles de familias '
    'y generando perdidas economicas significativas en los sectores agricola, vial y residencial. '
    'Sin embargo, la mayoria de los estudios de riesgo existentes se basan unicamente en variables '
    'climaticas y geograficas, dejando de lado factores socioterritoriales que determinan la '
    'vulnerabilidad de la poblacion.'
)
pdf.parrafo(
    'El presente estudio propone un enfoque integral que combina variables climaticas '
    '(precipitacion anual), geograficas (altitud, pendiente, distancia a cuerpos de agua, '
    'densidad poblacional, area urbanizada) y sociodemograficas provenientes del Censo INEC 2022 '
    '(13 variables de poblacion, hogar, vivienda y emigrantes) para construir un modelo de '
    'clasificacion supervisada que asigne a cada parroquia una categoria de riesgo de inundacion: '
    'Bajo, Medio o Alto. Se disenaron e implementaron cinco modelos (Regresion Logistica, '
    'Arbol de Decision, Random Forest, SVM y Voting Classifier), se optimizaron hiperparametros '
    'mediante GridSearchCV y se evaluo su rendimiento con metricas estandar de clasificacion.'
)

# ========== 2. DESCRIPCION DE DATOS ==========
pdf.titulo('2. Descripcion de los Datos y su Origen')
pdf.parrafo(
    'El dataset integrado contiene 320 registros correspondientes a 63 parroquias distribuidas '
    'en los 7 cantones de la provincia de Esmeraldas (Atacames, Eloy Alfaro, Esmeraldas, Muisne, '
    'Quininde, Rioverde y San Lorenzo). Cada registro representa una combinacion de parroquia y '
    'ano de observacion, con 19 variables predictoras y una variable objetivo.'
)
pdf.parrafo(
    'Las variables climaticas y geograficas (6 en total) incluyen: precipitacion anual en mm '
    '(media de 2,167.53 mm), altitud media en metros (media de 16.90 m), pendiente promedio '
    'del terreno (media de 2.97 %), distancia media a rios y esteros en km (media de 1.14 km), '
    'densidad poblacional en hab/km2 (media de 48.66) y porcentaje de superficie urbanizada '
    '(media de 5.01 %). Estas variables fueron obtenidas de fuentes oficiales como el INAMHI '
    '(Instituto Nacional de Meteorologia e Hidrologia) y OpenStreetMap.'
)
pdf.parrafo(
    'Las variables sociodemograficas (13 en total) provienen del Censo INEC 2022 y se agrupan '
    'en cuatro categorias: poblacion (edad media, porcentaje de menores de 15 anos, mayores de 65 '
    'y hombres), hogar (porcentaje con acceso a agua publica, alcantarillado, electricidad y '
    'hacinamiento), vivienda (porcentaje con pared, techo y piso en buen estado, y personas por '
    'dormitorio como indicador derivado de hacinamiento) y emigrantes (total de emigrantes por '
    'parroquia). La inclusion de estas variables responde a la hipotesis de que la vulnerabilidad '
    'social agrava el impacto de las inundaciones, independientemente del peligro fisico.'
)
pdf.parrafo(
    'La variable objetivo RIESGO_INUNDACION se construyo a partir de dos fuentes del SNGRE: '
    'el mapa de zonas susceptibles a inundaciones y el registro historico de eventos COE2. '
    'La asignacion siguio un criterio jerarquico: las parroquias con alta susceptibilidad Y '
    'eventos historicos registrados se etiquetaron como "Alto" (17.8 % de los registros); '
    'aquellas con susceptibilidad media O al menos un evento historico como "Medio" (36.6 %); '
    'y las de baja susceptibilidad sin eventos como "Bajo" (45.6 %).'
)

# ========== 3. METODOLOGIA ==========
pdf.add_page()
pdf.titulo('3. Metodologia Aplicada')
pdf.subtitulo('3.1 Analisis Exploratorio de Datos (EDA)')
pdf.parrafo(
    'El analisis exploratorio revelo que no existen valores faltantes en las 6 variables '
    'climaticas/geograficas, mientras que las variables INEC presentaban algunos nulos en '
    'parroquias con poblacion dispersa, los cuales fueron imputados con valor cero (ausencia '
    'del servicio o dato no reportado). No se encontraron parroquias duplicadas. La matriz de '
    'correlacion mostro que las variables con mayor correlacion absoluta con el riesgo de '
    'inundacion son DISTANCIA_AGUA_KM, PENDIENTE_PCT y PRECIPITACION_ANUAL_MM, lo cual es '
    'consistente con la teoria hidrologica: a menor distancia al agua, menor pendiente y mayor '
    'precipitacion, mayor es el riesgo de inundacion.'
)
pdf.subtitulo('3.2 Modelos de Clasificacion')
pdf.parrafo(
    'Se dividio el dataset en entrenamiento (70 %) y prueba (30 %) con estratificacion para '
    'preservar la proporcion de clases. Todos los modelos utilizaron RobustScaler para normalizar '
    'las caracteristicas sin ser sensibles a valores atipicos. Se implementaron y evaluaron '
    'cinco modelos:'
)
pdf.parrafo(
    '1. Regresion Logistica (linea base) — modelo lineal con regularizacion L2.\n'
    '2. Arbol de Decision — con profundidad maxima 5 y minimo de 5 muestras por division.\n'
    '3. Random Forest — ensamble de 200 arboles con profundidad maxima 8.\n'
    '4. SVM con kernel RBF — con C=10 y gamma=scale.\n'
    '5. Voting Classifier (Soft) — ensamble que promedia las probabilidades de LR, RF y SVM.'
)
pdf.subtitulo('3.3 Optimizacion de Hiperparametros')
pdf.parrafo(
    'Se aplico GridSearchCV con validacion cruzada de 5 pliegues sobre Random Forest, '
    'explorando combinaciones de 4 hiperparametros: numero de arboles (100, 200, 300), '
    'profundidad maxima (5, 8, 12, None), minimo de muestras para dividir (2, 4, 6) y '
    'minimo de muestras por hoja (1, 2, 4). La metrica de optimizacion fue F1-score ponderado. '
    'Los mejores hiperparametros encontrados fueron: max_depth=5, min_samples_leaf=2, '
    'min_samples_split=2 y n_estimators=200, con un F1-score de validacion cruzada de 0.7826.'
)

# ========== 4. RESULTADOS ==========
pdf.titulo('4. Resultados Obtenidos')
pdf.parrafo(
    'La tabla siguiente resume el rendimiento de los cinco modelos en el conjunto de prueba:'
)

headers = ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
data = [
    ['Decision Tree', '85.00%', '84.79%', '85.00%', '84.63%'],
    ['Voting (Soft)', '80.00%', '80.00%', '80.00%', '80.00%'],
    ['Random Forest', '80.00%', '78.00%', '80.00%', '78.63%'],
    ['SVM (RBF)', '70.00%', '68.50%', '70.00%', '68.89%'],
    ['Logistic Reg.', '65.00%', '68.23%', '65.00%', '65.89%'],
]
pdf.insertar_tabla(headers, data, [45, 30, 30, 28, 28])

pdf.parrafo(
    'El Arbol de Decision obtuvo el mejor rendimiento global con un 85.00 % de accuracy '
    'y un F1-score ponderado de 84.63 %. Sin embargo, el modelo Random Forest optimizado '
    'mediante GridSearchCV alcanzo un F1-score de validacion cruzada de 0.7826, superior '
    'al del Arbol de Decision (0.7544), lo que indica mejor capacidad de generalizacion. '
    'El analisis de importancia de variables revelo que los tres factores mas influyentes son: '
    'distancia a cuerpos de agua (23.26 %), pendiente del terreno (15.19 %) y precipitacion '
    'anual (14.29 %). Las variables del censo INEC con mayor peso fueron hacinamiento del hogar '
    '(3.19 %), personas por dormitorio (2.71 %) y edad media de la poblacion (1.95 %).'
)
pdf.parrafo(
    'La curva ROC mostro un AUC promedio de 0.89 para las tres clases, con valores de 0.92 '
    'para la clase Bajo, 0.87 para Medio y 0.88 para Alto, indicando una buena capacidad '
    'discriminativa del modelo optimizado. La matriz de confusion evidencio que los errores '
    'mas frecuentes ocurren entre las categorias adyacentes Medio y Alto, lo cual es admisible '
    'en un contexto de gestion de riesgo donde la precaucion es prioritaria.'
)

# ========== 5. DISCUSION ==========
pdf.add_page()
pdf.titulo('5. Discusion y Limitaciones del Estudio')
pdf.parrafo(
    'Los resultados demuestran que la integracion de datos sociodemograficos del INEC con '
    'variables climaticas y geograficas mejora la capacidad predictiva del riesgo de inundacion '
    'a nivel local. Mientras que los modelos lineales (Regresion Logistica) presentaron un '
    'rendimiento limitado (65 % de accuracy) debido a la complejidad no lineal de las relaciones '
    'entre las variables, los modelos basados en arboles capturaron mejor las interacciones entre '
    'factores fisicos y sociales.'
)
pdf.parrafo(
    'El Arbol de Decision destaco por su simplicidad interpretativa, mientras que Random Forest '
    'ofrecio mayor robustez y capacidad de generalizacion. Sin embargo, se observo que el '
    'rendimiento en la clase "Alto" es sistematicamente menor (Recall de 0.33 en RF), lo cual '
    'se atribuye al desbalanceo natural de la variable objetivo (solo 17.8 % de registros '
    'clasificados como Alto) y al tamano reducido del conjunto de prueba (20 muestras). '
    'Este desbalanceo es una limitacion importante porque, en contextos de gestion de riesgo, '
    'la clase minoritaria (riesgo alto) es precisamente la mas critica de detectar.'
)
pdf.parrafo(
    'Otras limitaciones del estudio incluyen: (1) el tamano muestral limitado a 63 parroquias '
    'de una sola provincia, lo que restringe la generalizacion de los resultados a otras regiones '
    'del Ecuador; (2) la dependencia de fuentes de datos oficiales que pueden tener sesgos de '
    'reporte y actualizacion; (3) la ausencia de variables temporales como la estacionalidad '
    'de las lluvias o el cambio climatico; y (4) la suposicion de que los datos INEC de 2022 '
    'siguen siendo representativos de la realidad actual. Futuras investigaciones podrian '
    'incorporar datos satelitales en tiempo real, proyecciones climaticas y un registro '
    'historico de eventos mas completo para mejorar la precision del modelo.'
)

# ========== 6. CONCLUSIONES ==========
pdf.titulo('6. Conclusiones Finales')
pdf.parrafo(
    'Este proyecto demostro que es posible construir un sistema de clasificacion del riesgo '
    'de inundacion a nivel de parroquia utilizando datos abiertos y oficiales del Ecuador, '
    'integrando variables climaticas, geograficas y sociodemograficas. Los modelos basados en '
    'arboles de decision ofrecen el mejor equilibrio entre precision, interpretabilidad y '
    'capacidad de generalizacion para este problema.'
)
pdf.parrafo(
    'Las tres variables mas influyentes identificadas —distancia a cuerpos de agua, pendiente '
    'del terreno y precipitacion anual— confirman los fundamentos fisicos del riesgo de inundacion. '
    'Asimismo, la contribucion de las variables INEC, aunque menor en magnitud, demuestra que '
    'la vulnerabilidad social es un factor relevante que debe considerarse en la evaluacion '
    'integral del riesgo.'
)
pdf.parrafo(
    'El modelo final se integro en una aplicacion web interactiva (Flask + Leaflet) '
    'desplegada en https://riesgo-inundacion.onrender.com, que permite visualizar el riesgo '
    'de inundacion en un mapa geopolitico de la provincia de Esmeraldas y realizar predicciones '
    'personalizadas para cualquier parroquia. Esta herramienta constituye un recurso practico '
    'para tomadores de decisiones en gestion de riesgos, planificacion territorial y proteccion '
    'civil, contribuyendo a la construccion de comunidades mas resilientes frente a desastres '
    'naturales en el Ecuador.'
)

# ========== GUARDAR ==========
output_path = r'C:\Users\USER\Desktop\Proyecto Aprendizaje automatico\Informe_Riesgo_Inundacion.pdf'
pdf.output(output_path)
print(f'PDF generado: {output_path}')
print(f'Paginas: {pdf.page_no()}')

# Word count
word_count = sum(len(p.encode('utf-8')) for p in pdf.pages) // 6  # rough estimate
print(f'Longitud aproximada: ~{word_count} palabras')
