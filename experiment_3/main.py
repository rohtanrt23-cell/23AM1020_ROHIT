from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import joblib

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, age: int = Form(...), salary: int = Form(...)):

    features = np.array([[age, salary]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        result = "User will purchase the product"
    else:
        result = "User will NOT purchase the product"

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "prediction": result}
    )