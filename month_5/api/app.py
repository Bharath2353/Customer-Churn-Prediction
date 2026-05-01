from fastapi import FastAPI
from src.predict import predict

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def get_prediction(data: dict):
    values = list(data.values())
    prob, pred = predict(values)
    return {"probability": prob, "churn": pred}