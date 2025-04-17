import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# URL de ton API Flask (à adapter selon ton cas) 
API_URL = "http://127.0.0.1:5000/predict"

st.title("Prédiction de défaut de crédit 💳")
st.write("Entrez un identifiant client pour voir la prédiction du modèle.")

# Champ de saisie
sk_id_curr = st.number_input("Entrez l'identifiant du client (SK_ID_CURR)", step=1)

# Bouton pour lancer la prédiction
if st.button("Prédire"):
    with st.spinner("Chargement de la prédiction..."):
        try:
            # Envoi de la requête POST
            response = requests.post(API_URL, json={"SK_ID_CURR": sk_id_curr})
            if response.status_code == 200:
                result = response.json()

                proba = result['probability']
                shap_values = result['shap_values']
                feature_names = result['feature_names']
                feature_values = result['feature_values']

                st.success(f"Probabilité de défaut : {proba:.2f}%")

                # Affichage des features et shap values
                df_shap = pd.DataFrame({
                    'Feature': feature_names,
                    'Valeur': feature_values,
                    'SHAP': shap_values
                }).sort_values("SHAP", key=abs, ascending=False)

                st.subheader("Importance des variables (SHAP)")
                st.dataframe(df_shap)

                # Petit graphe SHAP
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(df_shap['Feature'], df_shap['SHAP'], color='skyblue')
                ax.invert_yaxis()
                ax.set_title("Valeurs SHAP")
                st.pyplot(fig)

            else:
                st.error(f"Erreur {response.status_code} : {response.text}")

        except Exception as e:
            st.error(f"Une erreur s'est produite : {str(e)}")
