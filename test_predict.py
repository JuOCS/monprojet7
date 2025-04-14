import requests

# Adresse de ton API Flask (utilise 127.0.0.1 si tu es sur le même ordi)
url = "http://127.0.0.1:5000/predict"

# JSON avec un ID valide (remplace par un ID réel issu de df_reel)
data = {
    "SK_ID_CURR": 100001
}

# Envoi de la requête POST
response = requests.post(url, json=data)

# Affichage du résultat
if response.status_code == 200:
    print("✅ Réponse de l'API :")
    print(response.json())
else:
    print("❌ Erreur : ", response.status_code)
    print(response.text)