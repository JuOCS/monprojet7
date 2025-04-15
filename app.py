import os
import joblib
import pandas as pd
import shap
from flask import Flask, jsonify, request
from sklearn.preprocessing import StandardScaler
from azure.storage.blob import BlobServiceClient

# Initialiser  Flask
app = Flask(__name__)

# Connexion au stockage Azure
BLOB_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "my-container"

def download_blob(blob_name, save_path):
# Créer le dossier s'il n'existe pas
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
    blob_client = blob_service.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
    with open(save_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

# Télécharger les fichiers depuis Azure Blob Storage
download_blob("modele_pipeline.pkl", "data/modele_pipeline.pkl")
download_blob("dataframeP7.pkl", "data/dataframeP7.pkl")

# Charger les données
df = pd.read_pickle("data/dataframeP7.pkl")
df_reel = df[df["TARGET"].isna()]

# Charger le modèle
pipeline = joblib.load("data/modele_pipeline.pkl")
scaler = pipeline.named_steps['scaler']
model = pipeline.named_steps['classifier']

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.json
        sk_id_curr = data['SK_ID_CURR']

        sample = df_reel[df_reel['SK_ID_CURR'] == sk_id_curr]
        if sample.empty:
            return jsonify({'error': 'ID non trouvé dans les données'}), 404

        sample = sample.drop(columns=['TARGET'])
        sample_scaled = scaler.transform(sample)

        prediction = model.predict_proba(sample_scaled)
        proba = prediction[0][1] * 100

        # SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_scaled)[0][0].tolist()

        return jsonify({
            'probability': proba,
            'shap_values': shap_values,
            'feature_names': sample.columns.tolist(),
            'feature_values': sample.values[0].tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Lancer l'API
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
