# Predicción de Demanda Eléctrica a Corto Plazo - ML & DL

> **Evaluación comparativa de modelos de aprendizaje automático y aprendizaje profundo para la predicción de la demanda eléctrica a corto plazo: estudio de caso de Shahrekord**

*Comparative evaluation of machine learning and deep learning models for short-term electricity load forecasting: a case study of Shahrekord.*

Repositorio del proyecto final del curso **Sistemas Inteligentes y Machine Learning** (Ingeniería de Sistemas Computacionales, Universidad Privada del Norte).

## Autores

| Autor | Rol |
| --- | --- |
| Pedro Pablo Zavaleta Molina | Investigador principal / desarrollo del código |
| Andree Felix Casafranca Villano | Analista de datos |
| Kevin David Roman Quispe | Evaluación de modelos y análisis de resultados |
| Jose Antonio Ogosi Auqui | Docente del curso / supervisión metodológica |

## Descripción

Este proyecto compara experimentalmente cuatro modelos de predicción de la demanda eléctrica a la hora siguiente (`Load(t+1)`):

- **Random Forest** (ensemble de árboles)
- **SVR** (Support Vector Regression, kernel RBF)
- **XGBoost** (gradient boosting)
- **LSTM** (Long Short-Term Memory)

La demanda histórica, las variables meteorológicas y las temporales se organizaron en cuatro grupos de indicadores que dan lugar a cuatro escenarios experimentales:

| Escenario | Descripción | Variables |
| --- | --- | --- |
| A | Historial de demanda | 9 |
| B | Historial + temporales | 22 |
| C | Historial + meteorológicas | 27 |
| D | Modelo completo | 41 |

Cada escenario se optimiza de forma independiente (hiperparámetros) y se evalúa con **MAE**, **RMSE**, **R²** y **MAPE**, junto con el coste computacional.

## Dataset

- **Nombre:** [Shahrekord Electricity Demand Dataset](https://github.com/sardarimohsen/Shahrekord-Electricity-Demand-Dataset)
- **Ubicación:** Shahrekord, Irán
- **Resolución:** horaria (24 observaciones/día)
- **Observaciones:** 35 064 (4 años, marzo 2015 – marzo 2019)
- **Objetivo:** demanda eléctrica (MW), con 25 variables meteorológicas y temporales

## Resultados clave (escenario D — modelo completo)

| Modelo | MAE (MW) | RMSE (MW) | R² | MAPE (%) |
| --- | --- | --- | --- | --- |
| **XGBoost** | 3,90 | **6,48** | **0,9880** | 1,74 |
| Random Forest | **4,16** | 7,14 | 0,9854 | **1,85** |
| LSTM | 5,29 | 8,40 | 0,9798 | 2,31 |
| SVR | 7,71 | 10,90 | 0,9660 | 3,64 |

*Baseline de persistencia (Load(t+1) = Load(t)): MAE = 8,06 MW.*

### Validación temporal (origen móvil, 3 ventanas de prueba)

El ordenamiento de los modelos se mantiene estable entre ventanas: XGBoost es el más preciso y estable (R² = 0,985 ± 0,006), seguido de Random Forest, LSTM y SVR.

| Modelo | MAE (media ± desv.) | RMSE (media ± desv.) | R² (media ± desv.) |
| --- | --- | --- | --- |
| XGBoost | 3,21 ± 1,52 | 5,12 ± 2,54 | 0,985 ± 0,006 |
| Random Forest | 3,48 ± 1,75 | 5,81 ± 2,80 | 0,980 ± 0,007 |
| LSTM | 4,43 ± 2,40 | 6,26 ± 3,42 | 0,977 ± 0,010 |
| SVR | 8,65 ± 2,50 | 12,52 ± 4,16 | 0,871 ± 0,120 |

**Conclusiones principales:** los modelos de árboles (XGBoost y Random Forest) son los más competitivos. Las variables temporales aportan la mayor mejora sobre el historial de demanda, mientras que la meteorología por sí sola no la aporta (su efecto estacional ya queda absorbido por el calendario). XGBoost ofrece la mejor relación entre precisión y coste computacional.

## Estructura del repositorio

```
MachineLearning-G4/
├── 1-Article/          # Artículo (paper), informe final e informe de competencias
├── 2-Data/
│   ├── shahrekord/     # Dataset utilizado (94_95_96_97_all.csv)
│   └── dataset_anterior-SIN_USAR/
├── 3-Experiment/
│   ├── experiment.py   # Flujo reproducible completo
│   └── *.csv           # Resultados generados (results, escenarios, tiempos, etc.)
├── 4-Figures/          # Figuras generadas con Matplotlib
└── 5-Evidence/         # Capturas de evidencia (sección VIII) y RESULTS.txt
```

## Requisitos

- **Python 3.12** (el proyecto usa `py -3.12`; la 3.14 no tiene las librerías instaladas)
- Librerías: `numpy 2.5.2`, `pandas 3.0.5`, `scikit-learn 1.9.0`, `xgboost 3.4.1`, `tensorflow 2.21.0`, `matplotlib 3.11.1`

```bash
pip install numpy pandas scikit-learn xgboost tensorflow matplotlib
```

## Cómo ejecutar

Desde la raíz del repositorio:

```bash
py -3.12 3-Experiment/experiment.py
```

El script carga y limpia el dataset, construye las características, optimiza los hiperparámetros por escenario, entrena y evalúa los cuatro modelos, y genera los CSV de resultados y las figuras. Tarda entre 15 y 25 minutos (la LSTM es la parte más lenta).

## Entorno de referencia

- OS: Windows 11 · CPU: AMD Ryzen 5 5500U (6 núcleos / 12 hilos) · RAM: 16 GB · GPU: ninguna
- Semilla aleatoria fija: `42`

## Artículo e informe

Los documentos completos (artículo en formato DYNA/APA, informe final e informe de competencias) están disponibles en la carpeta `1-Article/`.
