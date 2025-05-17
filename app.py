import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.preprocessing import StandardScaler
from azure.storage.blob import BlobServiceClient
import traceback

app = Flask(__name__)

# === CONFIGURATION ===
BLOB_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "my-container"
DATA_DIR = "data"
MODEL_PATH = os.path.join(DATA_DIR, "modele_pipeline.pkl")
DF_PATH = os.path.join(DATA_DIR, "dataframeP7.pkl")

# === VARIABLES GLOBALES ===
pipeline = None
scaler = None
model = None
df_cache = None  # Cache du DataFrame

# === FONCTIONS UTILITAIRES ===
def download_blob(blob_name, save_path):
    if not BLOB_CONN_STR:
        raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING non défini.")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
    blob_client = blob_service.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
    with open(save_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

def load_model():
    global pipeline, scaler, model
    if pipeline is None:
        if not os.path.exists(MODEL_PATH):
            download_blob("modele_pipeline.pkl", MODEL_PATH)
        pipeline = joblib.load(MODEL_PATH)
        scaler = pipeline.named_steps['scaler']
        model = pipeline.named_steps['classifier']

def load_client_data(sk_id_curr):
    global df_cache
    if df_cache is None:
        if not os.path.exists(DF_PATH):
            download_blob("dataframeP7.pkl", DF_PATH)
        df_cache = pd.read_pickle(DF_PATH)
    # Filtrage sur l'ID
    df_reel = df_cache[(df_cache["TARGET"].isna()) & (df_cache["SK_ID_CURR"] == sk_id_curr)]
    return df_reel

# === ROUTES ===
@app.route("/")
def home():
    return jsonify({"message": "API opérationnelle"}), 200

@app.route("/predict", methods=['POST'])
def predict():
    try:
        load_model()

        data = request.json
        sk_id_curr = data.get('SK_ID_CURR')
        if sk_id_curr is None:
            return jsonify({'error': 'SK_ID_CURR requis'}), 400

        client_df = load_client_data(sk_id_curr)
        if client_df.empty:
            return jsonify({'error': 'ID non trouvé dans les données'}), 404

        sample = client_df.drop(columns=['TARGET'])
        sample_scaled = scaler.transform(sample)
        prediction = model.predict_proba(sample_scaled)
        proba = prediction[0][1] * 100

        return jsonify({
            'probability': proba,
            'feature_names': sample.columns.tolist(),
            'feature_values': sample.values[0].tolist()
        })

    except Exception:
        print(traceback.format_exc())  # Pour logs Render
        return jsonify({'error': 'Erreur interne lors de la prédiction'}), 500
