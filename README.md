# 📊 API de Predicción de Demanda — Trazos y Hojas

API REST que predice la demanda diaria de los productos de una papelería a partir
de su histórico de ventas, usando un modelo de Machine Learning (LightGBM).

El modelo se entrenó en un proyecto previo de análisis de datos; este repositorio
lo convierte en un **servicio accesible por internet**, con endpoints de predicción,
validación de datos y una interfaz web para probarlo.

🔗 **Demo en vivo:** [trazos-y-hojas-api.onrender.com](https://trazos-y-hojas-api.onrender.com)

![CI](https://github.com/terelopdel-sketch/Trazos_y_Hojas_Api/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)

---

## 🎯 El problema

La papelería **Trazos y Hojas** decide la reposición de inventario de forma intuitiva,
lo que provoca dos problemas: **roturas de stock** (ventas perdidas) y **exceso de
inventario** (capital inmovilizado).

Este proyecto ofrece una previsión objetiva de la demanda diaria por producto, para
apoyar esas decisiones con datos en lugar de intuición.

---

## 🚀 Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET`  | `/` | Página de inicio con formularios interactivos para probar la API |
| `GET`  | `/health` | Estado del servicio y del modelo |
| `GET`  | `/docs` | Documentación interactiva (Swagger), generada automáticamente |
| `POST` | `/predict` | Predicción de un día a partir de las variables ya calculadas |
| `POST` | `/predict-historico` | Predicción de un día a partir del histórico de ventas |
| `POST` | `/predict-semana` | Previsión a 7 días (predicción autoregresiva) |

### Ejemplos de uso

**Predicción de un día con las variables ya calculadas:**

```python
import requests

datos = {
    "lag_1": 5, "lag_7": 4, "media_movil_7": 4.5,
    "media_movil_14": 4.2, "std_movil_7": 1.3, "Dia_semana": 2
}
r = requests.post("https://trazos-y-hojas-api.onrender.com/predict", json=datos)
print(r.json())
# {'unidades_estimadas': 5.66, 'detalle': 'Demanda estimada para el dia...'}
```

**Predicción de un día a partir del histórico de ventas:**

```python
datos = {
    "ventas": [3, 5, 2, 4, 6, 1, 0, 4, 5, 3, 2, 4, 6, 5],
    "Dia_semana": 2
}
r = requests.post("https://trazos-y-hojas-api.onrender.com/predict-historico", json=datos)
print(r.json())
# {'unidades_estimadas': 5.02, 'detalle': 'Demanda estimada a partir de 14 dias...'}
```

**Previsión a 7 días:**

```python
r = requests.post("https://trazos-y-hojas-api.onrender.com/predict-semana", json=datos)
for dia in r.json()["predicciones"]:
    print(f"Día {dia['dia']}: {dia['unidades_estimadas']} unidades")
```

---

## 🧠 El modelo

- **Algoritmo:** LightGBM (gradient boosting)
- **Variables:** `lag_1`, `lag_7`, `media_movil_7`, `media_movil_14`, `std_movil_7`, `Dia_semana`
- **Transformación:** el target se entrena en escala `log1p` y se revierte con `expm1`
- **Selección del modelo:** se compararon varios algoritmos y conjuntos de variables
  con validación cruzada temporal, evaluando sobre datos completamente aislados.

### Métricas sobre el conjunto de test

| Métrica | Valor |
|---|---|
| RMSE | 23.99 |
| MAE | 4.05 |
| R² | 0.364 |

Un error absoluto medio de ~4 unidades por predicción, suficiente para apoyar las
decisiones de reposición en un problema de demanda intermitente.

---

## 📈 Predicción a 7 días (autoregresiva)

El endpoint `/predict-semana` genera la previsión encadenando predicciones: la
estimación de cada día se incorpora al histórico para predecir el siguiente.

> ⚠️ Como cada día se apoya en las predicciones de los anteriores, la fiabilidad
> disminuye con el horizonte: el día 7 es menos preciso que el día 1. Es un
> comportamiento esperado en la previsión a varios días con este tipo de modelos.

---

## 🛠️ Tecnologías

- **FastAPI** — framework de la API, con validación automática y documentación en `/docs`
- **LightGBM** — modelo de predicción
- **Pydantic** — validación de los datos de entrada
- **pytest** — batería de tests
- **GitHub Actions** — integración continua (los tests se ejecutan en cada push)
- **Render** — despliegue del servicio

---

## 🌐 Uso en producción (sin instalar nada)

La API está desplegada y accesible públicamente en Render. No necesitas instalar
nada: puedes usarla directamente desde el navegador o desde código.

🔗 **URL base:** [https://trazos-y-hojas-api.onrender.com](https://trazos-y-hojas-api.onrender.com)

- **Interfaz web:** abre la URL en el navegador para probar las predicciones con formularios.
- **Documentación interactiva:** [/docs](https://trazos-y-hojas-api.onrender.com/docs)
- **Desde código:** todos los ejemplos de este README ya apuntan a esta URL base.

> ℹ️ El servicio está alojado en el plan gratuito de Render. Si lleva un rato sin
> usarse, la primera petición puede tardar unos 30-60 segundos en responder mientras
> el servicio se reactiva. Las siguientes son inmediatas.

---

## 💻 Ejecución en local

```bash
# 1. Clonar el repositorio
git clone https://github.com/terelopdel-sketch/Trazos_y_Hojas_Api.git
cd Trazos_y_Hojas_Api

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash
# source venv/bin/activate        # Linux / Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Arrancar la API
uvicorn app.main:app --reload
```

La API queda disponible en `http://127.0.0.1:8000` y la documentación en
`http://127.0.0.1:8000/docs`.

### Ejecutar los tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

---

## 📁 Estructura del proyecto

````
Trazos_y_Hojas_Api/
├── app/
│   └── main.py
├── models/
│   ├── modelo_final_lgbm.joblib
│   └── features_modelo_final.joblib
├── tests/
│   └── test_api.py
├── .github/workflows/
│   └── ci.yml
├── requirements.txt
├── requirements-dev.txt
├── runtime.txt
└── render.yaml
````

---

## 👤 Autora

**Tere** — Proyecto personal de Machine Learning y despliegue de modelos.