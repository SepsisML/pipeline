from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv

from typing import Optional
import joblib
import numpy as np
import os


# Load the model
model = joblib.load('models/gbdt_model.pkl')
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
# Define input model for validation


class PredictionInput(BaseModel):
    _id: Optional[str] = None
    HR: float
    O2Sat: float
    Temp: float
    SBP: float
    MAP: float
    DBP: float
    Resp: float
    EtCO2: float
    BaseExcess: float
    HCO3: float
    FiO2: float
    pH: float
    PaCO2: float
    SaO2: float
    AST: float
    BUN: float
    Alkalinephos: float
    Calcium: float
    Chloride: float
    Creatinine: float
    Bilirubin_direct: float
    Glucose: float
    Lactate: float
    Magnesium: float
    Phosphate: float
    Potassium: float
    Bilirubin_total: float
    TroponinI: float
    Hct: float
    Hgb: float
    PTT: float
    WBC: float
    Fibrinogen: float
    Platelets: float
    Age: float
    Gender: float
    Unit1: float
    Unit2: float
    HospAdmTime: float
    ICULOS: float
    SepsisLabel: float
    Hora: float
    Paciente: str
    HospitalName: str
    Day: float


lab_attributes = [
    "pH", "PaCO2", "AST", "BUN", "Alkalinephos", "Chloride", "Creatinine",
    "Lactate", "Magnesium", "Potassium", "Bilirubin_total", "PTT", "WBC",
    "Fibrinogen", "Platelets"
]

vital_attributes = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"
]

expected_keys = lab_attributes + vital_attributes

app = FastAPI(title="Medical Prediction API")


@app.post("/predict", summary="Full Patient Prediction", description="Predict using full patient data")
async def predict(input_data: PredictionInput):
    try:
        # Convert input to numpy array
        features = [
            input_data.pH, input_data.PaCO2, input_data.AST, input_data.BUN, input_data.Alkalinephos,
            input_data.Chloride, input_data.Creatinine, input_data.Lactate, input_data.Magnesium,
            input_data.Potassium, input_data.Bilirubin_total, input_data.PTT, input_data.WBC,
            input_data.Fibrinogen, input_data.Platelets, input_data.HR, input_data.O2Sat,
            input_data.Temp, input_data.SBP, input_data.MAP, input_data.DBP, input_data.Resp,
        input_data.EtCO2
        ]




        # Asegurar que no haya datos faltantes
        if len(features) < len(expected_keys):
            raise HTTPException(status_code=400, detail="Faltan algunos features")

        # Convertir a numpy array y hacer la predicción
        features_array = np.array(features).reshape(1,-1)
        prediction = model.predict(features_array)

        return {
            "prediction": int(prediction[0]),
            "patient": input_data.Paciente,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PatientQueryModel(BaseModel):
    patient: str
    hour: str

@app.post("/predict/by-patient", 
         summary="Prediction by Patient and Hour", 
         description="Retrieve predictions for a specific patient at a given hour")
async def predict_by_patient(data: PatientQueryModel):
    try:
        client = MongoClient(MONGO_URI)
        db = client['SepsisTraining']
        collection = db['DataPacientesSource']        
        
        query = {
            "Paciente": data.patient, 
            "Hora": float(data.hour)
        }
        projection = {key: 1 for key in expected_keys}
        records = list(collection.find(query, projection))
        
        # If no records found, return a 404-like response
        if not records:
            return {
                "patient": data.patient,
                "hour": data.hour,
                "message": "No records found for this patient and hour",
                "record_count": 0
            }
        
        # Process records
        processed_records = []
        for record in records:
            try:
                features = [record.get(key, -9999) for key in expected_keys]
                
                features_array = np.array(features).reshape(1, -1)
                prediction_result = model.predict(features_array)
                processed_records.append(int(prediction_result[0]))
            except Exception as conversion_error:
                # Log the error, but continue processing other records
                print(f"Error processing record: {conversion_error}")
        
        return {
            "patient": data.patient,
            "hour": data.hour,
            "record_count": len(processed_records),
            "records": processed_records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))