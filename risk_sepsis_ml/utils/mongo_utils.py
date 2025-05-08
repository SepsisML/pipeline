# utils/mongo_utils.py
from pymongo import MongoClient


def write_collection(dataframe, collection_name, db_name='SepsisTraining', mongo_uri='mongodb://localhost:27017/'):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    collection.drop()
    print("El df a escribir es ", dataframe)
    collection.insert_many(dataframe.to_dict(orient='records'))
