from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score, confusion_matrix

import latam_airlines.utils.paths as path
rs = {'random_state': 42}

def check_quality(df):
    """Check the quality from data, count the NA, DUPLICATED and unique.
    Args:
        df (DataFrame): DataFrame to check
    """
    # Verificamos la cantidad de valores NA en cada columna
    print("Valores NA por columna:")
    print(df.isna().sum())

    # Verificamos la cantidad de datos duplicados en el dataframe
    print("\nCantidad de datos duplicados:")
    print(df.duplicated().sum())

    # Verificamos la cantidad de valores únicos en cada columna
    print("\nValores únicos por columna:")
    print(df.nunique())

# Crear la columna 'periodo_dia'
def categorized_days_periods(row):
    """ filter to classify the period of day
    Args:
        row (Serie): DataFrame row
    """
    hora = row['Fecha-I'].hour
    if hora >= 5 and hora <= 11:
        return 'mañana'
    elif hora >= 12 and hora <= 18:
        return 'tarde'
    else:
        return 'noche'

def rm_outliers(df, col):
    """Remove the outliers from data.
    Args:
        df (DataFrame): DataFrame to check
        col (str): Columna to remove outliers
    Return:
        df (DataFrame): DataFrame ok
    """

    p_05 = df[col].quantile(0.05) # 5th quantile
    p_95 = df[col].quantile(0.95) # 95th quantile
    df[col].clip(p_05, p_95, inplace=True)
    return df

# Classification - Model Pipeline
def modelPipeline(X_train, X_test, y_train, y_test):

    log_reg = LogisticRegression(**rs,solver='saga', max_iter=200)
    mlp = MLPClassifier(max_iter=500, **rs)
    dt = DecisionTreeClassifier(**rs)
    rf = RandomForestClassifier(**rs)
    xgb = XGBClassifier(**rs, verbosity=0)

    # Crear un objeto ColumnTransformer
    # Crear una pipeline que incluya el ColumnTransformer
    # Crear un objeto ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(with_mean=False), ['DIA', 'MES', 'HORA', 'MIN','Vlo_change', 'Emp_change','temporada_alta']),
            ('cat-nominal', OneHotEncoder(), ['periodo_dia','DIANOM', 'MESNOM','Des-I', 'TIPOVUELO', 'OPERA'])
        ],
        )

    clfs = [
            ('Logistic Regression', log_reg), 
            ('MLP', mlp), 
            ('Decision Tree', dt), 
            ('Random Forest', rf), 
            ('XGBoost', xgb)
            ]

    pipelines = []

    scores_df = pd.DataFrame(columns=['Model', 'F1_Score', 'Precision', 'Recall', 'Accuracy'])


    for clf_name, clf in clfs:

        pipeline = Pipeline(steps=[
                                   ('preprocessor', preprocessor),
                                   ('classifier', clf)
                                   ]
                            )
        pipeline.fit(X_train, y_train)


        y_pred = pipeline.predict(X_test)
        # F1-Score
        fscore = f1_score(y_test, y_pred,average='weighted')
        # Precision
        pres = precision_score(y_test, y_pred,average='weighted')
        # Recall
        rcall = recall_score(y_test, y_pred,average='weighted')
        # Accuracy
        accu = accuracy_score(y_test, y_pred)


        pipelines.append(pipeline)

        scores_df = scores_df.append({
                                    'Model' : clf_name, 
                                    'F1_Score' : fscore,
                                    'Precision' : pres,
                                    'Recall' : rcall,
                                    'Accuracy' : accu
                                    }, 
                                    ignore_index=True)
        
    return pipelines, scores_df

def update_model(model: Pipeline) -> None:
    """update or create model pkl version.
    Args:
        model(Pipeline): pipeline trained
    """
    dump(model, path.models_dir('model.pkl'))


def save_simple_metrics_report(train_score: float, test_score: float, report, model: Pipeline) -> None:
    """Create simple report with metrics of performance from Modelo or KPI's and models parameters.
    Args:
        train_score(float):score train
        test_score(float):score test
        validation_score(float): score val
        mse(float): RMSE
        mae(float): MAE
        mape(float): MAPE
        model(Pipeline): Model parameters
    """
    with open(path.reports_dir('report.txt'), 'w') as report_file:

        report_file.write('# Model Pipeline Description'+'\n')

        for key, value in model.named_steps.items():
            report_file.write(f'### {key}:{value.__repr__()}'+'\n')

        report_file.write(f'### Train Score: {train_score}'+'\n')
        report_file.write(f'### Test Score: {test_score}'+'\n')
        report_file.write(f'### Validation Score: {validation_score}'+'\n')
        report_file.write(f'### Reporte:{report}'+'\n')

def get_model_performance_test_set(y_real: pd.Series, y_pred: pd.Series) ->None:
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    sns.regplot(x=y_pred, y=y_real, ax = ax)
    ax.set_xlabel('Predicted worldwide gross')
    ax.set_ylabel('Real worldwide gross')
    ax.set_title('Behavior of model prediction')
    fig.savefig(path.reports_figures_dir('prediction_behavior.png'))