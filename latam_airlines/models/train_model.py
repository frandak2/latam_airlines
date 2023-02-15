# Importar las librerías necesarias
import latam_airlines.utils.paths as path
from latam_airlines.utils.logger import get_logging_config
from latam_airlines.utils.latam_utils import modelPipeline, get_model_performance_test_set, save_simple_metrics_report, update_model

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import logging.config
import warnings

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
rs = {'random_state': 42}

# configurando logging
logging.config.dictConfig(get_logging_config())

# Ignoring warnings
warnings.filterwarnings("ignore")


#Leemos los datos simulados
logging.info('Loading Data...')
data_train_balanced = pd.read_csv(path.data_processed_dir('data_train_balanced.csv'))

# Dividir los datos en variables de entrada y salida
logging.info('Split data...')
X = data_train_balanced.drop('atraso_15', axis=1)
y = data_train_balanced['atraso_15']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear una pipeline que incluya el ColumnTransformer y el RandomForestClassifier
# Crear un objeto ColumnTransformer
logging.info('Create Pipeline...')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler( with_mean=False), ['DIA', 'MES', 'HORA', 'MIN','Vlo_change', 'Emp_change','temporada_alta']),
        ('cat-nominal', OneHotEncoder(), ['periodo_dia','DIANOM', 'MESNOM','Des-I', 'TIPOVUELO', 'OPERA'])
    ],
    )
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model',RandomForestClassifier(**rs))
    ]
)

# realizar validacion cruzada
logging.info('Run CV...')
final_result = cross_validate(pipe, X_train, y_train, return_train_score=True, cv=5)

#observamos la dispersion del score
train_score_std = np.std(final_result['train_score'])
test_score_std = np.std(final_result['test_score'])
logging.info('Dispersion del score en la validacion cruzada')
logging.info(f'Dispersion train score: {train_score_std}')
logging.info(f'Dispersion test score: {test_score_std}')

# Observamos la media del score
train_score_mean = np.mean(final_result['train_score'])
test_score_mean = np.mean(final_result['test_score'])
logging.info('media del score en la validacion cruzada')
logging.info(f'media train score: {train_score_mean}')
logging.info(f'media test score: {test_score_mean}')

assert train_score_mean > 0.8
assert test_score_mean > 0.75

# Entrenar modelo
logging.info('Model training...')
pipe.fit(X_train, y_train)

# Mostrar los resultados
logging.info('Get metrics...')
validation_score =pipe.score(X_test, y_test)

y_pred = pipe.predict(X_test)

# Imprimiendo el reporte de clasificación
report = classification_report(y_test, y_pred)

# Crear reporte y metricas
get_model_performance_test_set('cm_balanced_over', y_test, y_pred)
save_simple_metrics_report('report_balanced_over', train_score_mean, test_score_mean, validation_score, report, pipe)
# Guardar modelo
logging.info('Save model...')
update_model('model_balanced_over',pipe)

logging.info('Model Training end...')
