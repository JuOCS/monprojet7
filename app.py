import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.preprocessing import StandardScaler
from azure.storage.blob import BlobServiceClient
# import shap  # À réactiver si besoin

app = Flask(__name__)

# Constantes
BLOB_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "my-container"
DATA_DIR = "data"
MODEL_PATH = os.path.join(DATA_DIR, "modele_pipeline.pkl")
DF_PATH = os.path.join(DATA_DIR, "dataframeP7.pkl")

# Fonction de téléchargement depuis Azure Blob Storage
def download_blob(blob_name, save_path):
    if not BLOB_CONN_STR:
        raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING non défini.")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
    blob_client = blob_service.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
    with open(save_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

# Route de test
@app.route("/")
def home():
    return jsonify({"message": "API en ligne"}), 200

# Route de prédiction
@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Télécharger fichiers si non présents
        if not os.path.exists(MODEL_PATH):
            download_blob("modele_pipeline.pkl", MODEL_PATH)
        if not os.path.exists(DF_PATH):
            download_blob("dataframeP7.pkl", DF_PATH)

        # Charger données et modèle à la demande
        df = pd.read_pickle(DF_PATH)
        df_reel = df[df["TARGET"].isna()]
        pipeline = joblib.load(MODEL_PATH)

        scaler = pipeline.named_steps['scaler']
        model = pipeline.named_steps['classifier']

        # Récupérer l'ID du client
        data = request.json
        sk_id_curr = data.get('SK_ID_CURR')
        if sk_id_curr is None:
            return jsonify({'error': 'SK_ID_CURR requis'}), 400

        sample = df_reel[df_reel['SK_ID_CURR'] == sk_id_curr]
        if sample.empty:
            return jsonify({'error': 'ID non trouvé dans les données'}), 404

        sample = sample.drop(columns=['TARGET'])
        sample_scaled = scaler.transform(sample)
        prediction = model.predict_proba(sample_scaled)
        proba = prediction[0][1] * 100

        # Interprétabilité (optionnelle)
        # explainer = shap.TreeExplainer(model)
        # shap_values = explainer.shap_values(sample_scaled)[0][0].tolist()

        return jsonify({
            'probability': proba,
            # 'shap_values': shap_values,
            'feature_names': sample.columns.tolist(),
            'feature_values': sample.values[0].tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
