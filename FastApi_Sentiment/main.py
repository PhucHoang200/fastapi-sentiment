from fastapi import FastAPI
from pydantic import BaseModel
from model_loader import sentiment_model

app = FastAPI(
    title="Vietnamese Sentiment API",
    version="1.0.0"
)

class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Vietnamese Sentiment API is running"}

@app.post("/predict")
def predict_sentiment(data: InputText):
    output = sentiment_model.predict(data.text)
    return output
