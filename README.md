# latam_airlines

# El problema
El problema consiste en predecir la probabilidad de atraso de los vuelos que aterrizan o despegan del aeropuerto de Santiago de Chile (SCL). Para eso les entregamos un dataset usando datos públicos y reales donde cada fila corresponde a un vuelo que aterrizó o despegó de SCL. Para cada vuelo se cuenta con la siguiente información:

Fecha-I : Fecha y hora programada del vuelo.
Vlo-I : Número de vuelo programado.
Ori-I : Código de ciudad de origen programado.
Des-I : Código de ciudad de destino programado.
Emp-I : Código aerolínea de vuelo programado.
Fecha-O : Fecha y hora de operación del vuelo.
Vlo-O : Número de vuelo de operación del vuelo.
Ori-O : Código de ciudad de origen de operación
Des-O : Código de ciudad de destino de operación.
Emp-O : Código aerolínea de vuelo operado.
DIA : Día del mes de operación del vuelo.
MES : Número de mes de operación del vuelo.
AÑO : Año de operación del vuelo.
DIANOM : Día de la semana de operación del vuelo.
TIPOVUELO : Tipo de vuelo, I =Internacional, N =Nacional.
OPERA : Nombre de aerolínea que opera.
SIGLAORI : Nombre ciudad origen.
SIGLADES : Nombre ciudad destino.

# Desafío
1. ¿Cómo se distribuyen los datos? ¿Qué te llama la atención o cuál es tu conclusión sobre esto?

2. Genera las columnas adicionales y luego expórtelas en un archivo synthetic_features.csv :
    - temporada_alta : 1 si Fecha-I está entre 15-Dic y 3-Mar, o 15-Jul y 31-Jul, o 11-Sep y 30-Sep, 0 si no.
    - dif_min : diferencia en minutos entre Fecha-O y Fecha-I .
    - atraso_15 : 1 si dif_min > 15, 0 si no.
    - periodo_dia : mañana (entre 5:00 y 11:59), tarde (entre 12:00 y 18:59) y noche (entre 19:00 y 4:59), en base a Fecha-I .

3. ¿Cómo se compone la tasa de atraso por destino, aerolínea, mes del año, día de la semana, temporada, tipo de vuelo? ¿Qué variables esperarías que más influyeran en predecir atrasos?

4. Entrena uno o varios modelos (usando el/los algoritmo(s) que prefieras) para estimar la probabilidad de atraso de un vuelo. Siéntete libre de generar variables adicionales y/o complementar con variables externas.

5. Evalúa tu modelo. ¿Qué performance tiene? ¿Qué métricas usaste para evaluar esa performance y por qué? ¿Por qué elegiste ese algoritmo en particular? ¿Qué variables son las que más influyen en la predicción? ¿Cómo podrías mejorar la performance? 

# Desarrollo del desafio
Con esta corta descripcion buscamos dar una idea general de como sera abordado el problema para predecir la probabilidad de atraso de vuelos en el aeropuerto de Santiago de Chile (SCL):

1. EDA
Análisis exploratorio de datos (EDA) para entender la distribución de los datos y conocer las características relevantes del dataset. Esto incluiría verificar la presencia de valores faltantes, outliers, distribución de las variables, graficas, etc.

2. Generación de columnas adicionales:
    - Crear una columna "temporada_alta" que identifique si el vuelo se encuentra dentro de los períodos de temporada alta.
    - Crear una columna "dif_min" que calcule la diferencia en minutos entre la hora programada y la hora de operación del vuelo.
    - Crear una columna "atraso_15" que identifique si el vuelo tuvo un atraso mayor a 15 minutos.
    - Crear una columna "periodo_dia" que identifique en qué período del día (mañana, tarde, noche) se programó el vuelo.

3. Análisis de tasa de atraso:
    - Calcular la tasa de atraso para cada destino, aerolínea, mes, día de la semana, temporada, y tipo de vuelo.
    - Analizar la relación entre las variables y la tasa de atraso para identificar cuáles son las variables que más influyen en la tasa de atraso.

4. Modelado:
    - Seleccionar uno o varios algoritmos de aprendizaje automático para entrenar el modelo.
    - Generar nuevas variables o combinar variables externas para mejorar la precisión del modelo.
    - Realizar validación cruzada para evaluar la performance del modelo y seleccionar los hiperparámetros adecuados.

5. Evaluación del modelo:
    - Medir la performance del modelo con métricas relevantes, como la exactitud, AUC-ROC, matriz de confusión, entre otras.
    - Analizar qué variables tienen más impacto en la predicción.
    - Identificar posibles áreas de mejora, como la inclusión de más variables, la modificación de la arquitectura del modelo, entre otras.

6. Crecion de API y Dockerfile
    - Creacion de main, models y views para desplegar una API REST que me regrese el valor predicho por el modelo
    - Testear la API
    - Crear un contenedor de la API usando Docker 

pd: podemos usar solo git para la creacion de ramas y versionamiento, podemos leer este articulo: [A modern branching strategy](https://martinfowler.com/articles/ship-show-ask.html)

## Installation guide

Please read [install.md](install.md) for details on how to set up this project.

## Project Organization

    ├── LICENSE
    ├── tasks.py           <- Invoke with commands like `notebook`.
    ├── README.md          <- The top-level README for Data Scientist using this project.
    ├── install.md         <- Detailed instructions to set up this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-fmh-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures         <- Generated graphics and figures to be used in reporting.
    │
    ├── environment.yml    <- The requirements file for reproducing the analysis environment.
    │
    ├── .here              <- File that will stop the search if none of the other criteria
    │                         apply when searching head of project.
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .)
    │                         so latam_airlines can be imported.
    │
    └── latam_airlines               <- Source code for use in this project.
        ├── __init__.py    <- Makes latam_airlines a Python module.
        │
        ├── data           <- Scripts to download or generate data.
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling.
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions.
        │   ├── predict_model.py
        │   └── train_model.py
        │
        ├── utils          <- Scripts to help with common tasks.
            └── paths.py   <- Helper functions to relative file referencing across project.
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations.
            └── visualize.py

---
Project based on the [cookiecutter conda data science project template](https://github.com/frandak2/cookiecutter-personal).