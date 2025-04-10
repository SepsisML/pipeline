# Optional: Set environment variables before running
export MONGO_URI='mongodb://localhost:27017'
export MONGO_DATABASE='SepsisTraining'
export MONGO_COLLECTION='DataPacientesSource'

# Run the FastAPI server
uvicorn main:app --reload