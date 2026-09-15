"""
API REST para el modelo de predicción de demanda - Trazos y Hojas
Se utiliza el modeo LightGBM entrenado sobre lista_4 (con transformación log1p).

Framework: FastAPI
Ejecución local: uvicorn app.main:app --reload
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELO_PATH = BASE_DIR / "Models" / "modelo_final_lgbm.joblib"
FEATURES_PATH = BASE_DIR / "Models" / "features_modelo_final.joblib"

app = FastAPI(
    title="API Predicción de Demanda - Trazos y Hojas",
    description="Predice la demanda diaria de un producto de la papelería a partir de su histórico reciente.",
    version="1.0.0",
)

# ----------------------------------------------------------------------
# Carga del modelo (una sola vez, al arrancar)
# ----------------------------------------------------------------------
modelo = joblib.load(MODELO_PATH)
features = joblib.load(FEATURES_PATH)   # orden exacto de columnas que espera el modelo

# ----------------------------------------------------------------------
# Esquema de entrada — Pydantic valida los datos automáticamente.
# Si falta un campo o el tipo es incorrecto, FastAPI responde 422 con
# un mensaje claro, sin llegar a tocar el modelo.
# ----------------------------------------------------------------------
class DatosProducto(BaseModel):
    lag_1: float = Field(..., ge=0, description="Unidades vendidas el día anterior")
    lag_7: float = Field(..., ge=0, description="Unidades vendidas hace 7 días")
    media_movil_7: float = Field(..., ge=0, description="Media de ventas de los últimos 7 días")
    media_movil_14: float = Field(..., ge=0, description="Media de ventas de los últimos 14 días")
    std_movil_7: float = Field(..., ge=0, description="Desviación típica de ventas de los últimos 7 días")
    Dia_semana: int = Field(..., ge=0, le=6, description="Día de la semana (0=lunes ... 6=domingo)")

    model_config = {
        "json_schema_extra":{
            "example": {
                "lag_1": 5, "lag_7": 4, "media_movil_7": 4.5,
                "media_movil_14": 4.2, "std_movil_7": 1.3, "Dia_semana": 2
            }
        }
    }


class Prediccion(BaseModel):
    unidades_estimadas: float
    detalle: str


# ----------------------------------------------------------------------
# Endpoint "/" — landing page: explica cómo usar la API
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def landing():
    return """
    <html>
      <head>
        <meta charset="utf-8">
        <title>API Predicción de Demanda — Trazos y Hojas</title>
        <style>
          body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto;
                 padding: 0 20px; color: #333; line-height: 1.6; }
          h1 { color: #4A6035; }
          code { background: #EFE6D2; padding: 2px 6px; border-radius: 4px; }
          pre  { background: #EFE6D2; padding: 14px; border-radius: 8px; overflow-x: auto; }
          .endpoint { border-left: 4px solid #D9824A; padding-left: 14px; margin: 22px 0; }
        </style>
      </head>
      <body>
        <h1>API de Predicción de Demanda</h1>
        <p>Papelería <strong>Trazos y Hojas</strong>. Estima la demanda diaria de un
        producto a partir de su histórico reciente de ventas.</p>

        <div class="endpoint">
          <h3>POST <code>/predict</code></h3>
          <p>Devuelve la predicción de unidades. Espera un JSON con estos campos:</p>
          <pre>{
  "lag_1": 5,
  "lag_7": 4,
  "media_movil_7": 4.5,
  "media_movil_14": 4.2,
  "std_movil_7": 1.3,
  "Dia_semana": 2
}</pre>
        </div>

        <div class="endpoint">
          <h3>GET <code>/health</code></h3>
          <p>Comprueba que el servicio está activo y el modelo cargado.</p>
        </div>

        <div class="endpoint">
          <h3>GET <code>/docs</code></h3>
          <p>Documentación interactiva (Swagger): permite probar los endpoints
          desde el navegador.</p>
        </div>

        <p style="margin-top:30px; color:#8F7C52;">
          Modelo: LightGBM · variables lag_1, lag_7, media_movil_7, media_movil_14,
          std_movil_7, Dia_semana.
        </p>
      </body>
    </html>
    """


# ----------------------------------------------------------------------
# Endpoint /health — comprobación de estado
# ----------------------------------------------------------------------
@app.get("/health")
def health():
    return{"estado": "ok", "modelo_cargado": modelo is not None, "n_features": len(features)}


# ----------------------------------------------------------------------
# Endpoint /predict — la predicción del modelo
# ----------------------------------------------------------------------
@app.post("/predict", response_model=Prediccion)
def predict(datos: DatosProducto):
    try:
        # Construir un DataFrame en el ORDEN EXACTO de features que espera el modelo
        entrada = pd.DataFrame([[getattr(datos, f) for f in features]], columns=features)

        # El modelo predice en escala log1p -> revertir con expm1 y recortar a 0
        pred_log = modelo.predict(entrada)[0]
        pred = float(np.clip(np.expm1(pred_log), 0, None))

        return Prediccion(
            unidades_estimadas=round(pred, 2),
            detalle=f"Demanda estimada para el día de la semana {datos.Dia_semana}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error el general la predicción: {e}")



# ----------------------------------------------------------------------
# TERCER ENDPOINT (comentado) — para el redespliegue en directo.
# Descomentar durante la presentación, hacer commit y redesplegar.
# Devuelve qué variables usa el modelo y en qué orden.
# ----------------------------------------------------------------------
@app.get("/model-info")
def model_info():
     return {
         "algoritmo": "LightGBM",
         "n_variables": len(features),
         "variables": list(features),
         "transformacion_target": "log1p (revertida con expm1)",
     }