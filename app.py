import streamlit as st
import pandas as pd
import requests
import json
from io import BytesIO

# URL de ton modèle MLflow local
MLFLOW_URL = "http://127.0.0.1:5001/invocations" 

st.title("🧠 Prédictions via modèle MLflow (Conteneur Docker en local)")

# Liste des features (adaptée à ton dataset, ici exemple avec le dataset creditcard.csv)
feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Onglets pour le choix du mode
tab1, tab2 = st.tabs(["🔍 Prédiction simple", "📂 Prédiction par lots (CSV)"])

# --- MODE 1 : prédiction sur un seul point de données
with tab1:
    st.header("Prédiction sur un point de données")

    # Widgets dynamiques
    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Faire la prédiction"):
        df = pd.DataFrame([user_input])
        payload = {"dataframe_split": df.to_dict(orient="split")}
        response = requests.post(MLFLOW_URL, json=payload)

        if response.status_code == 200:
            prediction = response.json()['predictions'][0]
            st.success(f"✅ Prédiction : {prediction}")
        else:
            st.error(f"Erreur : {response.status_code} — {response.text}")

# --- MODE 2 : prédiction par lots via CSV
with tab2:
    st.header("Prédiction sur plusieurs données (batch CSV)")

    uploaded_file = st.file_uploader("Charge ton fichier CSV avec les colonnes de features", type=["csv"])

    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)

        st.subheader("Aperçu des données :")
        st.dataframe(df_batch.head())

        if st.button("Faire les prédictions sur tout le fichier"):
            payload = {"dataframe_split": df_batch.to_dict(orient="split")}
            response = requests.post(MLFLOW_URL, json=payload)

            if response.status_code == 200:
                preds = response.json()['predictions']
                df_batch["prediction"] = preds
                st.success("✅ Prédictions réalisées")

                st.dataframe(df_batch)

                # Télécharger le fichier
                output = BytesIO()
                df_batch.to_csv(output, index=False)
                st.download_button(
                    label="📥 Télécharger le fichier avec prédictions",
                    data=output.getvalue(),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"Erreur : {response.status_code} — {response.text}")
