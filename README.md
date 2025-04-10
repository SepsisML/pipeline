# 🔍 Sepsis Machine Learning Models with FastAPI & AWS Lambda

Este proyecto tiene como objetivo **disponibilizar modelos predictivos de sepsis** de dos maneras:

1. A través de una API REST construida con **FastAPI**.
2. Mediante una función **serverless** desplegada en **AWS Lambda**.

Ambas soluciones permiten realizar inferencias a partir de entradas clínicas (`X`) y obtener predicciones de riesgo de sepsis. Además, se incluye un sistema de pruebas para validar los modelos.

---

## 📦 Componentes del Proyecto

### 1. `fast-api`
Una API REST desarrollada con [FastAPI](https://fastapi.tiangolo.com/) que permite disponibilizar los modelos de predicción de sepsis como un servicio web.

- 🌐 Alternativa flexible al despliegue serverless.
- 🔧 Útil tanto para entornos de desarrollo como para producción auto-gestionada.
- ⚡ Alta velocidad de respuesta y bajo consumo de recursos.

> Endpoint típico:  
> `POST /predict`  
> con datos JSON representando las características de entrada del modelo (`X`).

---

### 2. `lambda-test`
Script de prueba diseñado para evaluar el rendimiento y la precisión de cada modelo.

- 🧪 Carga modelos desde archivos locales o endpoints.
- 📊 Evalúa la predicción para entradas específicas `X`.
- ✅ Útil para pruebas unitarias, regresión y validación previa al despliegue.

> Uso común:
> ```bash
> python lambda-test.py --model model.pkl --input sample_input.json
> ```

---

### 3. `lambda-api`
Función predictiva desplegada en AWS utilizando el entorno *Serverless (Lambda)*.

- ☁️ Permite escalar la inferencia en producción sin necesidad de gestionar infraestructura.
- 🛠️ Compatible con API Gateway para exposición de endpoints HTTP.
- 🧬 Utiliza los mismos modelos que FastAPI, garantizando coherencia en las predicciones.

> Ejemplo de uso (request HTTP):
> ```http
> POST https://z4y112sq8e.execute-api.us-east-1.amazonaws.com/dev/predict
> Content-Type: application/json
>
> {
>   "feature1": value1,
>   "feature2": value2,
>   ...
> }
> ```

---

## 🚀 Requisitos

- Python 3.8+
- FastAPI
- Uvicorn (para correr `fast-api`)
- AWS CLI y framework `serverless` (para desplegar `lambda-api`)

---

## 🛠️ Instalación y Uso

### Ejecutar la API con FastAPI (`fast-api`):
```bash
uvicorn fast-api.main:app --reload
