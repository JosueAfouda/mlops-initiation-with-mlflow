from fastapi import FastAPI, Request
import mlflow.pyfunc
import pandas as pd

# Charger le modèle MLflow (exporté avec `mlflow models save`)
model = mlflow.pyfunc.load_model("rf_model")  # ou chemin relatif/absolu

app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    input_json = await request.json()
    try:
        # Créer un DataFrame pandas à partir du JSON envoyé
        df = pd.DataFrame(input_json["data"], columns=input_json["columns"])
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}
