# utils/mongo_utils.py
from pymongo import MongoClient
from config import MONGO_URI, DEFAULT_DB_NAME

def write_collection(dataframe, collection_name, db_name=DEFAULT_DB_NAME, mongo_uri=MONGO_URI):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    collection.drop()
    collection.insert_many(dataframe.to_dict(orient='records'))
