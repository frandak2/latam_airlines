from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_no_atraso_prediction():
    response = client.post('/v1/prediction', json = {
                                                    'DIA': 1,
                                                    'MES': 1,
                                                    'HORA': 0,
                                                    'MIN': 15,
                                                    'Vlo_change': 0,
                                                    'Emp_change': 0,
                                                    'temporada_alta': 1,
                                                    'periodo_dia': 'noche',
                                                    'DIANOM': 'Domingo',
                                                    'MESNOM': 'Ene',
                                                    'Des_I': 'MMMX',
                                                    'TIPOVUELO': 'I',
                                                    'OPERA': 'Aeromexico'
                                                })
    assert response.status_code == 200
    assert response.json()['atraso_15'] == 0 

def test_atraso_prediction():
    response = client.post('/v1/prediction', json = {
                                                    'DIA': 1,
                                                    'MES': 1,
                                                    'HORA': 7,
                                                    'MIN': 35,
                                                    'Vlo_change': 0,
                                                    'Emp_change': 1,
                                                    'temporada_alta': 1,
                                                    'periodo_dia': 'mañana',
                                                    'DIANOM': 'Domingo',
                                                    'MESNOM': 'Ene',
                                                    'Des_I': 'SPJC',
                                                    'TIPOVUELO': 'I',
                                                    'OPERA': 'Grupo LATAM'
                                                })
    assert response.status_code == 200
    assert response.json()['atraso_15'] == 1




