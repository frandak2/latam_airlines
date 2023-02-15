from joblib import load
from scipy.sparse import data
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from pandas import DataFrame
import os 
from io import BytesIO
import latam_airlines.utils.paths as path

def get_model() -> Pipeline:
    model = load(path.models_dir('model_balanced_over.pkl'))
    return model

def transform_to_dataframe(class_model: BaseModel) -> DataFrame:
    transition_dictionary = {key:[value] for key, value in class_model.dict().items()}
    data_frame = DataFrame(transition_dictionary)
    return data_frame