from pymongo import MongoClient
import pandas as pd
import json
import os

# Obtener la ruta del directorio actual
directorio_actual = os.getcwd()


def mongoimport(csv_path, db_name, coll_name, db_url='localhost'):
    """
    Importa archivos CSV a MongoDB.

    :param csv_path: Ruta del archivo CSV.
    :param db_name: Nombre de la base de datos en MongoDB.
    :param coll_name: Nombre de la colección en MongoDB.
    :param db_url: URL del servidor de MongoDB.
    """
    client = MongoClient(db_url)
    db = client[db_name]
    coll = db[coll_name]

    data = pd.read_csv(csv_path, sep=',')
    payload = json.loads(data.to_json(orient='records'))
    coll.insert_many(payload)


# Definir la ruta de entrada basada en el directorio actual
ruta_entrada = os.path.join(directorio_actual, "output_data")

for paciente in os.listdir(ruta_entrada):
    archivo = os.path.join(ruta_entrada, paciente)

    if os.path.isfile(archivo) and archivo.endswith(".csv"):
        ndb = "SepsisTraining"
        col = "DataPacientesSource"
        try:
            mongoimport(archivo, ndb, col)
            print(f"Importado: {paciente}")
        except Exception as e:
            print(f"Se produjo un error con el paciente {paciente}: {e}")

print("Datos importados con éxito")
