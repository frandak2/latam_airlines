from pydantic import BaseModel

class PredictionRequest(BaseModel):
    DIA: int
    MES : int
    HORA: int
    MIN: int
    Vlo_change: int
    Emp_change: int
    temporada_alta: int
    periodo_dia: str
    DIANOM: str
    MESNOM: str
    Des_I: str
    TIPOVUELO: str
    OPERA: str

class PredictionResponse(BaseModel):
    atraso_15: int