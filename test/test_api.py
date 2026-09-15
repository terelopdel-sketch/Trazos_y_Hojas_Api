"""
Tests de la API de despliegue (Trazos y Hojas),

Estructura en 3 bloques:
    1. Llamada al endpoint de inicio ("/"), que explica la API.
    2. Llamada al endpoint de comprobacion de estado ("/health").
    3. Llamadas al endpoint de prediccion ("/predict"), con 3 supuestos:
       datos correctos, falta un campo, y un dato fuera de rango.

No reimplementamos ninguna validacion aqui (eso ya lo hace FastAPI + Pydantic
en main.py, con los Field(...) de DatosProducto). Este archivo solo hace
llamadas HTTP con distintos datos y comprueba que la API responde como se
espera en cada caso.

"""

import os
import requests

BASE_URL = ("https://app-trazos-y-hojas-0q2n.onrender.com")

# Mismo ejemplo que Tere puso en el esquema Pydantic (DatosProducto.model_config),
# asi nos aseguramos de que es un caso que el propio esquema considera valido.
INPUT_VALIDO = {
    "lag_1": 5,
    "lag_7": 4,
    "media_movil_7": 4.5,
    "media_movil_14": 4.2,
    "std_movil_7": 1.3,
    "Dia_semana": 2,
}


# ====================================================================
# 1. Endpoint de inicio ("/") - explica la API
# ====================================================================
print("\n--- 1. GET / (landing page) ---")
resp = requests.get(f"{BASE_URL}/")
print("Status:", resp.status_code, "(esperado 200)")


# ====================================================================
# 2. Endpoint de comprobacion de estado ("/health")
# ====================================================================
print("\n--- 2. GET /health ---")
resp = requests.get(f"{BASE_URL}/health")
print("Status:", resp.status_code, "(esperado 200)")
print("Respuesta:", resp.json())


# ====================================================================
# 3. Endpoint de prediccion ("/predict") - 3 supuestos
# ====================================================================

print("\n--- 3.1 POST /predict - datos correctos ---")
resp = requests.post(f"{BASE_URL}/predict", json=INPUT_VALIDO)
print("Status:", resp.status_code, "(esperado 200)")
print("Respuesta:", resp.json())

print("\n--- 3.2 POST /predict - falta un campo (lag_1) ---")
input_incompleto = INPUT_VALIDO.copy()
del input_incompleto["lag_1"]
resp = requests.post(f"{BASE_URL}/predict", json=input_incompleto)
print("Status:", resp.status_code, "(esperado 422)")

print("\n--- 3.3 POST /predict - dato fuera de rango (Dia_semana=9) ---")
input_fuera_de_rango = INPUT_VALIDO.copy()
input_fuera_de_rango["Dia_semana"] = 9
resp = requests.post(f"{BASE_URL}/predict", json=input_fuera_de_rango)
print("Status:", resp.status_code, "(esperado 422)")



