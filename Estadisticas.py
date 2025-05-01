from pymongo import MongoClient
from datetime import datetime
import statistics

# Configuración
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "SepsisTraining"
COLLECTION_NAME = "DataPacientesSource"

# Conexión
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


variable = "PTT"
# Consulta: obtener HR == -9999, ordenados por hora
cursor = collection.find({variable: {"$ne": -9999}}
                         ).sort([("Paciente", 1), ("Hora", 1)])

# Agrupar por paciente y calcular duración
tiempos_por_paciente = {}
anterior = {}

for doc in cursor:
    paciente = doc["Paciente"]
    hora = doc["Hora"]

    if paciente in anterior:
        duracion = hora - anterior[paciente]
        tiempos_por_paciente.setdefault(paciente, []).append(duracion)
    anterior[paciente] = hora

# Estadísticas
duraciones = [statistics.mean(v) for v in tiempos_por_paciente.values() if v]

if duraciones:
    print(
        f"Pacientes con al menos un valor faltante de {variable}: {len(duraciones)}")
    print(
        f"Tiempo promedio sin valores de {variable}: {statistics.mean(duraciones)} horas")
    print(
        f"Máximo: {max(duraciones)} max - Mínimo: {min(duraciones)} min")
else:
    print("No hay datos válidos para analizar.")
