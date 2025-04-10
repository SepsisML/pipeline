import json
import joblib
import numpy as np

# Cargar el modelo
try:
    model = joblib.load('gbdt_model.pkl')
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    exit()

# Datos de prueba (simulación de datos de entrada)
test_data = {
    "_id": "67d30791484e77f80d842d64",
    "HR": 75,
    "O2Sat": 98,
    "Temp": 0,
    "SBP": 165,
    "MAP": 100,
    "DBP": 71,
    "Resp": 19,
    "EtCO2": 0,
    "BaseExcess": -9999,
    "HCO3": -9999,
    "FiO2": -9999,
    "pH": 7.45,
    "PaCO2": 39,
    "SaO2": -9999,
    "AST": 36,
    "BUN": 32,
    "Alkalinephos": 42,
    "Calcium": -9999,
    "Chloride": 110,
    "Creatinine": 0.7,
    "Bilirubin_direct": -9999,
    "Glucose": -9999,
    "Lactate": 0,
    "Magnesium": 2.3,
    "Phosphate": -9999,
    "Potassium": 3.95,
    "Bilirubin_total": 0,
    "TroponinI": -9999,
    "Hct": -9999,
    "Hgb": -9999,
    "PTT": 0,
    "WBC": 5.7,
    "Fibrinogen": 0,
    "Platelets": 125,
    "Age": 56.99,
    "Gender": 1,
    "Unit1": -9999,
    "Unit2": -9999,
    "HospAdmTime": -0.02,
    "ICULOS": 141,
    "SepsisLabel": 1,
    "Hora": 140,
    "Paciente": "p000373",
    "HospitalName": "HospA",
    "Day": 6
}


expected_keys = [
    "pH", "PaCO2", "AST", "BUN", "Alkalinephos", "Chloride", "Creatinine",
    "Lactate", "Magnesium", "Potassium", "Bilirubin_total", "PTT", "WBC",
    "Fibrinogen", "Platelets", "HR", "O2Sat",
    "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]

# Extraer valores en el orden correcto
features = [test_data[key] for key in expected_keys]

# Convertir a array numpy
features_np = np.array(features).reshape(1, -1)

# Hacer la predicción
try:
    prediction = model.predict(features_np)
    print(f"🔮 Predicción del modelo: {prediction[0]}")
except Exception as e:
    print(f"❌ Error al hacer la predicción: {e}")
