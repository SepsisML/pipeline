import pandas as pd
import os

# Obtener la ruta del directorio actual
directorio_actual = os.getcwd() + "/training_setA"
hospital_name = "HospA"


def listar_directorio(ruta_origen, ruta_destino):
    """
    Convierte archivos PSV a CSV, añadiendo columnas de Paciente y Hora.

    :param ruta_origen: Directorio donde se encuentran los archivos PSV.
    :param ruta_destino: Directorio donde se guardarán los archivos CSV.
    """
    if not os.path.exists(ruta_destino):
        os.makedirs(ruta_destino)

    for archivo in os.listdir(ruta_origen):
        ruta_archivo = os.path.join(ruta_origen, archivo)

        if os.path.isfile(ruta_archivo) and archivo.endswith(".psv"):
            data1 = pd.read_table(ruta_archivo, sep='|')
            data = data1.fillna(-9999)

            nombre = os.path.splitext(archivo)[0]
            hora = 1

            for i in data.index:
                data.loc[i, "Hora"] = hora
                data.loc[i, "Paciente"] = nombre
                data.loc[i, "HospitalName"] = hospital_name
                hora += 1

            print(f"Procesando: {archivo}")

            ruta_salida = os.path.join(ruta_destino, f"{nombre}.csv")
            data.to_csv(ruta_salida, sep=',', index=False)
            print(f"Guardado: {ruta_salida}")


# Definir rutas basadas en el directorio actual
ruta_entrada = directorio_actual  # Carpeta actual
ruta_salida = os.path.join(
    os.getcwd(), "output_data")  # Carpeta de salida

listar_directorio(ruta_entrada, ruta_salida)
print("Conversión completada.")
