# utils/mongo_utils.py
from pymongo import MongoClient
from config import MONGO_URI, DEFAULT_DB_NAME

def write_collection(dataframe: pd.DataFrame, mongo_uri:str, db_name:str, collection_name:str):
        df_clean = dataframe.replace({np.nan: None})

        records = df_clean.to_dict(orient="records")

        if not records:
            print("DataFrame vacío. No se insertó nada.")
            return

        with MongoClient(mongo_uri) as client:
            db = client[db_name]
            collection = db[collection_name]
            collection.insert_many(records, ordered=False)


def load_collection(mongo_uri: str, db_name: str, collection_name: str):
    with MongoClient(mongo_uri) as client:
        db = client[db_name]
        collection = db[collection_name]

        data = list(collection.find({}, {"_id": 0}))

    if not data:
        print("La colección está vacía.")
        return pd.DataFrame()

    return pd.DataFrame(data)
