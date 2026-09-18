"""
API REST para el modelo de predicción de demanda — Trazos y Hojas
Sirve el modelo LightGBM entrenado sobre lista_4 (con transformación log1p).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELO_PATH = BASE_DIR / "models" / "modelo_final_lgbm.joblib"
FEATURES_PATH = BASE_DIR / "models" / "features_modelo_final.joblib"

app = FastAPI(
    title="API Predicción de Demanda — Trazos y Hojas",
    description="Predice la demanda diaria de un producto de la papelería a partir de su histórico reciente.",
    version="2.0.0",
)

# ----------------------------------------------------------------------
# Carga del modelo (una sola vez, al arrancar)
# ----------------------------------------------------------------------
modelo = joblib.load(MODELO_PATH)
features = joblib.load(FEATURES_PATH)   # orden exacto de columnas que espera el modelo


# ----------------------------------------------------------------------
# Esquemas de entrada / salida
# ----------------------------------------------------------------------
class DatosProducto(BaseModel):
    lag_1: float = Field(..., ge=0, description="Unidades vendidas el día anterior")
    lag_7: float = Field(..., ge=0, description="Unidades vendidas hace 7 días")
    media_movil_7: float = Field(..., ge=0, description="Media de ventas de los últimos 7 días")
    media_movil_14: float = Field(..., ge=0, description="Media de ventas de los últimos 14 días")
    std_movil_7: float = Field(..., ge=0, description="Desviación típica de ventas de los últimos 7 días")
    Dia_semana: int = Field(..., ge=0, le=6, description="Día de la semana (0=lunes … 6=domingo)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "lag_1": 5, "lag_7": 4, "media_movil_7": 4.5,
                "media_movil_14": 4.2, "std_movil_7": 1.3, "Dia_semana": 2
            }
        }
    }


class Prediccion(BaseModel):
    unidades_estimadas: float
    detalle: str


class DatosHistorico(BaseModel):
    """Entrada de los endpoints por histórico: la lista de ventas recientes."""
    ventas: list[float] = Field(
        ..., min_length=14,
        description="Ventas diarias recientes (al menos 14 días, el más antiguo primero)"
    )
    Dia_semana: int = Field(..., ge=0, le=6, description="Día de la semana a predecir (0=lunes … 6=domingo)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "ventas": [3, 5, 2, 4, 6, 1, 0, 4, 5, 3, 2, 4, 6, 5],
                "Dia_semana": 2
            }
        }
    }


class PrediccionDia(BaseModel):
    dia: int
    dia_semana: int
    unidades_estimadas: float


class PrediccionSemana(BaseModel):
    predicciones: list[PrediccionDia]
    detalle: str


# ----------------------------------------------------------------------
# Endpoint "/" — landing page con formulario interactivo
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def landing():
    return """
    <html>
      <head>
        <meta charset="utf-8">
        <title>API Prediccion de Demanda - Trazos y Hojas</title>
        <style>
          body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto;
                 padding: 0 20px; color: #333; line-height: 1.6; }
          h1 { color: #4A6035; }
          h2 { color: #4A6035; margin-top: 34px; font-size: 1.15em; }
          code { background: #EFE6D2; padding: 2px 6px; border-radius: 4px; }
          .endpoint { border-left: 4px solid #D9824A; padding-left: 14px; margin: 22px 0; }
          .botones { margin: 26px 0; }
          .btn { display: inline-block; padding: 12px 22px; border-radius: 8px;
                 text-decoration: none; color: white; font-weight: 600; margin: 0 10px 10px 0;
                 border: none; cursor: pointer; font-size: 1em; }
          .btn-verde  { background: #4A6035; }
          .btn-marron { background: #8F7C52; }
          .btn:hover { opacity: 0.9; }
          .formulario { background: #F7F4EC; border-radius: 10px; padding: 22px 24px; margin: 20px 0; }
          .campos { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px 20px; }
          .campo label { display: block; font-size: 0.9em; color: #4A6035; font-weight: 600;
                         margin-bottom: 4px; }
          .campo input, .campo select { width: 100%; padding: 8px 10px; border: 1px solid #CFC7B0;
                         border-radius: 6px; font-size: 1em; box-sizing: border-box; }
                    .grafica { display: flex; align-items: flex-end; gap: 10px; height: 220px;
                     margin-top: 20px; padding: 14px 10px 0; border-bottom: 2px solid #CFC7B0; }
          .barra-col { flex: 1; display: flex; flex-direction: column; align-items: center;
                       justify-content: flex-end; height: 100%; }
          .barra { width: 100%; background: #7D955B; border-radius: 6px 6px 0 0;
                   transition: height 0.3s; min-height: 2px; }
          .barra-valor { font-size: 0.82em; font-weight: 600; color: #4A6035; margin-bottom: 4px; }
          .barra-dia { font-size: 0.78em; color: #8F7C52; margin-top: 6px; text-align: center; }
          #previsionResultado { display: none; margin-top: 8px; }
          #previsionResultado .nota { font-size: 0.85em; color: #8F7C52; font-style: italic;
                     margin-top: 14px; }
          #resultado { margin-top: 18px; padding: 16px; border-radius: 8px; font-size: 1.05em;
                       display: none; }
          #resultado.ok  { background: #E3ECD8; border: 1px solid #7D955B; color: #33471F; }
          #resultado.err { background: #F6E2D8; border: 1px solid #D9824A; color: #8A3B18; }
                    #estado { display: none; margin: 4px 0 10px; padding: 14px 18px; border-radius: 8px;
                    background: #E3ECD8; border: 1px solid #7D955B; color: #33471F; }
                    #resultadoDia { margin-top: 18px; padding: 16px; border-radius: 8px; font-size: 1.05em;
                         display: none; }
          #resultadoDia.ok  { background: #E3ECD8; border: 1px solid #7D955B; color: #33471F; }
          #resultadoDia.err { background: #F6E2D8; border: 1px solid #D9824A; color: #8A3B18; }
          #estado.err { background: #F6E2D8; border-color: #D9824A; color: #8A3B18; }
          #estado .titulo { font-weight: 700; font-size: 1.05em; }
          #estado .detalle { font-size: 0.9em; color: #4A6035; margin-top: 4px; }
        </style>
      </head>
      <body>
        <h1>API de Prediccion de Demanda</h1>
        <p>Papeleria <strong>Trazos y Hojas</strong>. Estima la demanda diaria de un
        producto a partir de su historico reciente de ventas.</p>

        <div class="botones">
          <a class="btn btn-verde" href="/docs">Documentacion (/docs)</a>
          <button class="btn btn-marron" onclick="verEstado()">Estado del servicio</button>
        </div>

        <div id="estado"></div>

        <h2>Probar una prediccion</h2>
        <p>Rellena el historico reciente de un producto y pulsa <strong>Predecir</strong>:</p>

        <div class="formulario">
          <div class="campos">
            <div class="campo">
              <label>lag_1 (ventas ayer)</label>
              <input type="number" id="lag_1" value="5" step="any" min="0">
            </div>
            <div class="campo">
              <label>lag_7 (ventas hace 7 dias)</label>
              <input type="number" id="lag_7" value="4" step="any" min="0">
            </div>
            <div class="campo">
              <label>media_movil_7</label>
              <input type="number" id="media_movil_7" value="4.5" step="any" min="0">
            </div>
            <div class="campo">
              <label>media_movil_14</label>
              <input type="number" id="media_movil_14" value="4.2" step="any" min="0">
            </div>
            <div class="campo">
              <label>std_movil_7 (volatilidad)</label>
              <input type="number" id="std_movil_7" value="1.3" step="any" min="0">
            </div>
            <div class="campo">
              <label>Dia_semana (0=lunes ... 6=domingo)</label>
              <select id="Dia_semana">
                <option value="0">0 - Lunes</option>
                <option value="1">1 - Martes</option>
                <option value="2" selected>2 - Miercoles</option>
                <option value="3">3 - Jueves</option>
                <option value="4">4 - Viernes</option>
                <option value="5">5 - Sabado</option>
                <option value="6">6 - Domingo</option>
              </select>
            </div>
          </div>
          <div style="margin-top:18px;">
            <button class="btn btn-verde" onclick="predecir()">Predecir</button>
          </div>
          <div id="resultado"></div>
        </div>

                <h2>Predicción de un día (desde el histórico)</h2>
        <p>Introduce el histórico de ventas de un producto (al menos 14 días, separados
        por comas) y el día de la semana a predecir:</p>

        <div class="formulario">
          <div class="campo">
            <label>Histórico de ventas (más antiguo primero)</label>
            <input type="text" id="ventas_dia" value="3, 5, 2, 4, 6, 1, 0, 4, 5, 3, 2, 4, 6, 5">
          </div>
          <div class="campo" style="margin-top:14px; max-width:340px;">
            <label>Día de la semana a predecir</label>
            <select id="dia_dia">
              <option value="0">0 - Lunes</option>
              <option value="1">1 - Martes</option>
              <option value="2" selected>2 - Miercoles</option>
              <option value="3">3 - Jueves</option>
              <option value="4">4 - Viernes</option>
              <option value="5">5 - Sabado</option>
              <option value="6">6 - Domingo</option>
            </select>
          </div>
          <div style="margin-top:18px;">
            <button class="btn btn-verde" onclick="predecirDia()">Predecir un dia</button>
          </div>
          <div id="resultadoDia"></div>
        </div>

        <h2>Previsión a 7 días</h2>
        <p>Introduce el histórico de ventas (al menos 14 días, separados por comas)
        y el día de la semana del primer día a prever:</p>

        <div class="formulario">
          <div class="campo">
            <label>Histórico de ventas (más antiguo primero)</label>
            <input type="text" id="ventas" value="3, 5, 2, 4, 6, 1, 0, 4, 5, 3, 2, 4, 6, 5">
          </div>
          <div class="campo" style="margin-top:14px; max-width:340px;">
            <label>Día de la semana del primer día</label>
            <select id="dia_inicio">
              <option value="0">0 - Lunes</option>
              <option value="1">1 - Martes</option>
              <option value="2" selected>2 - Miercoles</option>
              <option value="3">3 - Jueves</option>
              <option value="4">4 - Viernes</option>
              <option value="5">5 - Sabado</option>
              <option value="6">6 - Domingo</option>
            </select>
          </div>
          <div style="margin-top:18px;">
            <button class="btn btn-verde" onclick="preverSemana()">Prever 7 dias</button>
          </div>
          <div id="previsionResultado">
            <div class="grafica" id="grafica"></div>
            <div class="nota" id="notaPrevision"></div>
          </div>
        </div>

        <div class="endpoint">
          <h3>Los endpoints de la API</h3>
          <p><code>POST /predict</code> - prediccion de un dia con variables ya calculadas.<br>
             <code>POST /predict-historico</code> - prediccion de un dia a partir del historico de ventas.<br>
             <code>POST /predict-semana</code> - prevision de 7 dias.<br>
             <code>GET /health</code> - comprueba que el servicio esta activo.<br>
             <code>GET /docs</code> - documentacion interactiva.</p>
        </div>

        <p style="margin-top:30px; color:#8F7C52;">
          Modelo: LightGBM - variables lag_1, lag_7, media_movil_7, media_movil_14,
          std_movil_7, Dia_semana.
        </p>

        <script>
          async function predecir() {
            const datos = {
              lag_1:          parseFloat(document.getElementById("lag_1").value),
              lag_7:          parseFloat(document.getElementById("lag_7").value),
              media_movil_7:  parseFloat(document.getElementById("media_movil_7").value),
              media_movil_14: parseFloat(document.getElementById("media_movil_14").value),
              std_movil_7:    parseFloat(document.getElementById("std_movil_7").value),
              Dia_semana:     parseInt(document.getElementById("Dia_semana").value)
            };
            const caja = document.getElementById("resultado");
            caja.style.display = "block";
            caja.className = "";
            caja.textContent = "Calculando...";
            try {
              const resp = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(datos)
              });
              const data = await resp.json();
              if (resp.ok) {
                caja.className = "ok";
                caja.innerHTML = "<strong>Demanda estimada: " +
                  data.unidades_estimadas + " unidades</strong><br>" + data.detalle;
              } else {
                caja.className = "err";
                caja.textContent = "Error en los datos enviados (revisa los campos).";
              }
            } catch (e) {
              caja.className = "err";
              caja.textContent = "No se pudo contactar con la API.";
            }
          }
                    async function predecirDia() {
            const texto = document.getElementById("ventas_dia").value;
            const ventas = texto.split(",").map(x => parseFloat(x.trim())).filter(x => !isNaN(x));
            const diaSemana = parseInt(document.getElementById("dia_dia").value);
            const caja = document.getElementById("resultadoDia");
            caja.style.display = "block";
            caja.className = "";
            caja.textContent = "Calculando...";

            if (ventas.length < 14) {
              caja.className = "err";
              caja.textContent = "Se necesitan al menos 14 dias de historico.";
              return;
            }
            try {
              const resp = await fetch("/predict-historico", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ventas: ventas, Dia_semana: diaSemana })
              });
              const data = await resp.json();
              if (resp.ok) {
                caja.className = "ok";
                caja.innerHTML = "<strong>Demanda estimada: " +
                  data.unidades_estimadas + " unidades</strong><br>" + data.detalle;
              } else {
                caja.className = "err";
                caja.textContent = "Error en los datos enviados.";
              }
            } catch (e) {
              caja.className = "err";
              caja.textContent = "No se pudo contactar con la API.";
            }
          }
                  const NOMBRES_DIA = ["Lun","Mar","Mie","Jue","Vie","Sab","Dom"];

          async function preverSemana() {
            const texto = document.getElementById("ventas").value;
            const ventas = texto.split(",").map(x => parseFloat(x.trim())).filter(x => !isNaN(x));
            const diaInicio = parseInt(document.getElementById("dia_inicio").value);
            const caja = document.getElementById("previsionResultado");
            const grafica = document.getElementById("grafica");
            const nota = document.getElementById("notaPrevision");

            if (ventas.length < 14) {
              caja.style.display = "block";
              grafica.innerHTML = "";
              nota.textContent = "Se necesitan al menos 14 dias de historico.";
              return;
            }

            try {
              const resp = await fetch("/predict-semana", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ventas: ventas, Dia_semana: diaInicio })
              });
              const data = await resp.json();
              if (!resp.ok) {
                caja.style.display = "block";
                grafica.innerHTML = "";
                nota.textContent = "Error en los datos enviados.";
                return;
              }
              const preds = data.predicciones;
              const maximo = Math.max(...preds.map(p => p.unidades_estimadas), 1);

              grafica.innerHTML = preds.map(p => {
                const altura = (p.unidades_estimadas / maximo) * 100;
                return '<div class="barra-col">' +
                         '<div class="barra-valor">' + p.unidades_estimadas + '</div>' +
                         '<div class="barra" style="height:' + altura + '%"></div>' +
                         '<div class="barra-dia">Dia ' + p.dia + '<br>' +
                            NOMBRES_DIA[p.dia_semana] + '</div>' +
                       '</div>';
              }).join("");

              nota.textContent = data.detalle;
              caja.style.display = "block";
            } catch (e) {
              caja.style.display = "block";
              grafica.innerHTML = "";
              nota.textContent = "No se pudo contactar con la API.";
            }
          }
                  async function verEstado() {
            const caja = document.getElementById("estado");
            caja.style.display = "block";
            caja.className = "";
            caja.innerHTML = "Comprobando...";
            try {
              const resp = await fetch("/health");
              const data = await resp.json();
              if (resp.ok && data.estado === "ok") {
                caja.className = "";
                caja.innerHTML =
                  '<div class="titulo">&#10003; Servicio activo</div>' +
                  '<div class="detalle">Modelo cargado correctamente &middot; ' +
                  data.n_features + ' variables de entrada</div>';
              } else {
                caja.className = "err";
                caja.innerHTML = '<div class="titulo">Servicio con problemas</div>';
              }
            } catch (e) {
              caja.className = "err";
              caja.innerHTML = '<div class="titulo">No se pudo contactar con el servicio</div>';
            }
          }
        </script>
      </body>
    </html>
    """


# ----------------------------------------------------------------------
# Endpoint /health — comprobación de estado
# ----------------------------------------------------------------------
@app.get("/health")
def health():
    return {"estado": "ok", "modelo_cargado": modelo is not None, "n_features": len(features)}


# ----------------------------------------------------------------------
# Endpoint /predict — predicción de un día con variables ya calculadas
# ----------------------------------------------------------------------
@app.post("/predict", response_model=Prediccion)
def predict(datos: DatosProducto):
    try:
        entrada = pd.DataFrame([[getattr(datos, f) for f in features]], columns=features)
        pred_log = modelo.predict(entrada)[0]
        pred = float(np.clip(np.expm1(pred_log), 0, None))
        return Prediccion(
            unidades_estimadas=round(pred, 2),
            detalle=f"Demanda estimada para el dia de la semana {datos.Dia_semana}.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar la prediccion: {e}")


# ----------------------------------------------------------------------
# Cálculo de features a partir del histórico de ventas
# ----------------------------------------------------------------------
def calcular_features(ventas, dia_semana):
    """A partir de una lista de ventas (>=14 días), calcula las 6 variables
    que espera el modelo, en el orden correcto."""
    v = np.array(ventas, dtype=float)
    fila = {
        "lag_1":          v[-1],
        "lag_7":          v[-7],
        "media_movil_7":  v[-7:].mean(),
        "media_movil_14": v[-14:].mean(),
        "std_movil_7":    v[-7:].std(ddof=1),
        "Dia_semana":     dia_semana,
    }
    return pd.DataFrame([[fila[f] for f in features]], columns=features)


def predecir_un_dia(ventas, dia_semana):
    """Predice las unidades de un día a partir del histórico (revierte log1p)."""
    entrada = calcular_features(ventas, dia_semana)
    pred_log = modelo.predict(entrada)[0]
    return float(np.clip(np.expm1(pred_log), 0, None))


# ----------------------------------------------------------------------
# Endpoint /predict-historico — predice UN día a partir del histórico
# ----------------------------------------------------------------------
@app.post("/predict-historico", response_model=Prediccion)
def predict_historico(datos: DatosHistorico):
    try:
        pred = predecir_un_dia(datos.ventas, datos.Dia_semana)
        return Prediccion(
            unidades_estimadas=round(pred, 2),
            detalle=f"Demanda estimada a partir de {len(datos.ventas)} dias de historico.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar la prediccion: {e}")


# ----------------------------------------------------------------------
# Endpoint /predict-semana — predice 7 días encadenados (autoregresivo)
# ----------------------------------------------------------------------
@app.post("/predict-semana", response_model=PrediccionSemana)
def predict_semana(datos: DatosHistorico):
    try:
        historico = list(datos.ventas)
        dia_semana = datos.Dia_semana
        predicciones = []

        for i in range(7):
            pred = predecir_un_dia(historico, dia_semana)
            predicciones.append(PrediccionDia(
                dia=i + 1,
                dia_semana=dia_semana,
                unidades_estimadas=round(pred, 2),
            ))
            historico.append(pred)
            dia_semana = (dia_semana + 1) % 7

        return PrediccionSemana(
            predicciones=predicciones,
            detalle="Prevision a 7 dias. La fiabilidad disminuye con el horizonte, "
                    "ya que cada dia se apoya en las predicciones de los anteriores.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar la prevision: {e}")