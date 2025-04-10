import json
import joblib
import numpy as np

# Cargar el modelo en el "cold start" de la función Lambda.
# Nota: Puedes empaquetar el modelo junto al código o utilizar Layers.
# Asegúrate de que la ruta sea correcta
model = joblib.load('gbdt_model.pkl')


def lambda_handler(event, context):
    try:
        # Extraer el cuerpo de la petición, que viene en formato JSON
        body = json.loads(event.get('body', '{}'))

        expected_keys = [
            "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "EtCO2",
            "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
            "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
            "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
            "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
            "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
            "Hora",
        ]

        # Extraer los valores en el orden definido
        features = [body.get(key) if body.get(
            key) is not None else -9999 for key in expected_keys]

        # Convertir a array numpy y preprocesar si es necesario
        features_np = np.array(features).reshape(1, -1)

        # Realizar la predicción
        prediction = model.predict(features_np)

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": float(prediction[0])})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
