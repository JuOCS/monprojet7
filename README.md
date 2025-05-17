# API Flask - Prédiction de Défaut de Crédit 💳

Cette API Flask expose une route `/predict` qui permet de prédire la probabilité de défaut de crédit pour un client identifié par son `SK_ID_CURR`. Elle utilise un pipeline de machine learning sérialisé, stocké dans **Azure Blob Storage**, et un DataFrame de données clients.

---

## 🔧 Fonctionnalités

- Téléchargement automatique du modèle (`modele_pipeline.pkl`) et des données (`dataframeP7.pkl`) depuis Azure Blob Storage
- Traitement du client à partir de son identifiant
- Retour JSON avec :
  - La probabilité de défaut (`probability`)
  - Les valeurs des variables utilisées (`feature_values`)
  - Les noms des variables (`feature_names`)

---

## 🧪 Exemple de requête

```http
POST /predict
Content-Type: application/json

{
  "SK_ID_CURR": 123456
}
