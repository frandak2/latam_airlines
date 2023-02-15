# importar librerias
##generacion de path
import latam_airlines.utils.paths as path
## funciones propias
from latam_airlines.utils.latam_utils import check_quality, categorized_days_periods
## manipulacion de datos
import pandas as pd
import numpy as np
## Visualización de los datos
import matplotlib.pyplot as plt
import seaborn as sns

# leer dataset

dataset_SCL = pd.read_csv(path.data_raw_dir('dataset_SCL.csv'), low_memory=False)


# Eliminamos valores NA
dataset_SCL.dropna(inplace=True)
# Eliminamos columnas que no aportan informacion
dataset_SCL.drop(columns=['Ori-I','Ori-O','SIGLAORI'], inplace=True)

# Valores unicos y ordenados por columna
dict_DesO = set(dataset_SCL['Des-O'])
dict_DesI = set(dataset_SCL['Des-I'])

# Extraemos las diferencias entre las 2 columnas
only_in_DesO = dict_DesO - dict_DesI
only_in_DesI = dict_DesI - dict_DesO

# Extraemos solo lo qu ecomparten entre las 2 columnas
both = dict_DesO & dict_DesI

# mostramos que hay de diferente
print(f"Elements only in DesO: {only_in_DesO}")
print(f"Elements only in DesI: {only_in_DesI}")
print(f"Len elements both: {len(both)}")

# Validación para ver si lo programado es igual a lo operado
dataset_SCL['Vlo_change'] = dataset_SCL.apply(lambda x: '0' if x['Vlo-I']==x['Vlo-O'] else '1', axis=1)
dataset_SCL['Emp_change'] = dataset_SCL.apply(lambda x:'0' if x['Emp-I']==x['Emp-O'] else '1', axis=1)

#Cambiamos el formato a fecha
cols_date = ['Fecha-I','Fecha-O']
for col in cols_date:
    dataset_SCL[col] = pd.to_datetime(dataset_SCL[col], format='%Y-%m-%d %H:%M:%S')

# creamos variables para ayudarnos a hacer el analisis
days = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
months = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
d = dict(zip(np.arange(1, 13),months))
dataset_SCL['MESNOM'] = dataset_SCL['MES'].astype(int).replace(d, regex=True)

#cambiamos las demas variables a formato
cols = ['Vlo-I', 'Des-I', 'Emp-I', 'Vlo-O', 'Des-O','Emp-O', 'DIA', 'MES', 'AÑO', 'DIANOM', 'TIPOVUELO', 'OPERA','SIGLADES', 'Vlo_change','Emp_change']
for col in cols:
    if col == 'DIANOM':
        dataset_SCL[col] = pd.Categorical(dataset_SCL[col],ordered=True, categories=days)
    else:
        dataset_SCL[col] = dataset_SCL[col].astype('category')

dataset_SCL.sort_values('Fecha-I', inplace=True)

# Crear la columna 'dif_min'
dataset_SCL['dif_min'] = (dataset_SCL['Fecha-O'] - dataset_SCL['Fecha-I']).dt.total_seconds() / 60# numero de segundos en un minuto

# Crear la columna 'atraso_15'
dataset_SCL['atraso_15'] =dataset_SCL['dif_min'].map(lambda x: '1' if x>15 else '0')

# creacion de mascara
temporada_alta = ((dataset_SCL['Fecha-I']>='2017-12-15') | (dataset_SCL['Fecha-I']<='2017-03-04')) | \
    ((dataset_SCL['Fecha-I']>='2017-07-15') & (dataset_SCL['Fecha-I']<='2017-08-01')) | \
        ((dataset_SCL['Fecha-I']>='2017-09-11') & (dataset_SCL['Fecha-I']<='2017-10-01'))

# podemos hacerlo con .loc tambien 
dataset_SCL['temporada_alta'] = np.where(temporada_alta, '1', '0')

# Crear la columna 'periodo_dia'
dataset_SCL['periodo_dia'] = dataset_SCL.apply(categorized_days_periods, axis=1)

dataset_SCL['HORA']=dataset_SCL['Fecha-I'].dt.hour
dataset_SCL['MIN']=dataset_SCL['Fecha-I'].dt.minute

# Crear una tabla de contingencia
top_destinos = pd.crosstab(dataset_SCL['Des-I'], dataset_SCL['atraso_15']).reset_index()

# Calcualo del total de vuelos por destino
top_destinos['total'] = top_destinos['0'] + top_destinos['1']

# filtrar por cantidad de vuelos al anio
DesI_filter = top_destinos[top_destinos['total']<=12]['Des-I'].unique()
dataset_SCL = dataset_SCL[~dataset_SCL['Des-I'].isin(DesI_filter)]

# Exportar las columnas adicionales a un archivo .csv
dataset_SCL.to_csv(path.data_processed_dir('synthetic_features.csv'), index=False)

# seleccion de cols para training
data_train = dataset_SCL[['DIA', 'MES', 'HORA', 'MIN', 'periodo_dia', 'DIANOM', 'MESNOM','Des-I', 'TIPOVUELO', 'OPERA', 'Vlo_change', 'Emp_change', 'temporada_alta', 'atraso_15']]

# Exportar el dataset para training en un .csv
data_train.to_csv(path.data_processed_dir('data_train.csv'), index=False)