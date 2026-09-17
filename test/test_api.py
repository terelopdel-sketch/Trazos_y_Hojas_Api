"""
Tests de la API con pytest.  Se lanzan con:  pytest
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

FEATURES_VALIDAS = {
    "lag_1": 5, "lag_7": 4, "media_movil_7": 4.5,
    "media_movil_14": 4.2, "std_movil_7": 1.3, "Dia_semana": 2,
}
HISTORICO_VALIDO = {
    "ventas": [3, 5, 2, 4, 6, 1, 0, 4, 5, 3, 2, 4, 6, 5],
    "Dia_semana": 2,
}


def test_landing_ok():
    r = client.get("/")
    assert r.status_code == 200
    assert "<h1>" in r.text


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"


def test_predict_ok():
    r = client.post("/predict", json=FEATURES_VALIDAS)
    assert r.status_code == 200
    assert r.json()["unidades_estimadas"] >= 0


def test_predict_falta_campo():
    datos = FEATURES_VALIDAS.copy()
    del datos["lag_1"]
    assert client.post("/predict", json=datos).status_code == 422


def test_predict_dia_fuera_de_rango():
    datos = {**FEATURES_VALIDAS, "Dia_semana": 9}
    assert client.post("/predict", json=datos).status_code == 422


def test_predict_historico_ok():
    r = client.post("/predict-historico", json=HISTORICO_VALIDO)
    assert r.status_code == 200
    assert r.json()["unidades_estimadas"] >= 0


def test_predict_historico_corto():
    datos = {"ventas": [1, 2, 3], "Dia_semana": 2}
    assert client.post("/predict-historico", json=datos).status_code == 422


def test_predict_semana_devuelve_7_dias():
    r = client.post("/predict-semana", json=HISTORICO_VALIDO)
    assert r.status_code == 200
    predicciones = r.json()["predicciones"]
    assert len(predicciones) == 7
    assert all(p["unidades_estimadas"] >= 0 for p in predicciones)


def test_predict_semana_avanza_dia_semana():
    r = client.post("/predict-semana", json=HISTORICO_VALIDO)
    dows = [p["dia_semana"] for p in r.json()["predicciones"]]
    assert dows == [2, 3, 4, 5, 6, 0, 1]


