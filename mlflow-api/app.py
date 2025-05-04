from fastapi import FastAPI, Request
import mlflow.pyfunc
import pandas as pd
import os

# Crée l'application FastAPI
app = FastAPI()

# Charger le modèle MLflow
MODEL_PATH = "rf_model"  # Le modèle doit contenir MLmodel, model.pkl, etc.

if not os.path.exists(os.path.join(MODEL_PATH, "MLmodel")):
    raise FileNotFoundError(f"Le fichier MLmodel est introuvable dans {MODEL_PATH}. "
                            "Assurez-vous que le modèle MLflow est bien copié dans l'image Docker.")

model = mlflow.pyfunc.load_model(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "API de prédiction avec MLflow et FastAPI"}

@app.post("/predict")
async def predict(request: Request):
    try:
        input_json = await request.json()
        data = input_json.get("data")
        columns = input_json.get("columns")

        if not data or not columns:
            return {"error": "JSON doit contenir 'data' et 'columns'"}

        df = pd.DataFrame(data, columns=columns)
        preds = model.predict(df)
        return {"predictions": preds.tolist()}

    except Exception as e:
        return {"error": str(e)}
