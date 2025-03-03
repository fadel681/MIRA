import streamlit as st
import pandas as pd
from model_utils import load_model, predict
from PIL import Image
import os

# Charger le modèle
model = load_model()

vars_sessions = ["prediction_score", "prediction_status", "prediction_model", "prediction_proba"]
for v in vars_sessions:
    if v not in st.session_state:
        st.session_state[v] = None

# Charger le logo
logo_path = os.path.join('media', 'Capgemini-Symbol.png')
logo = Image.open(logo_path)

# Afficher le logo réduit à gauche
st.sidebar.image(logo, use_container_width=True, width=100)

# Appliquer le CSS pour styliser l'application

# En-tête de l'application
st.markdown("# Predictive Tools of Credit Scoring")
   

# Conteneur principal
st.markdown("<div class='stFormContainer'>", unsafe_allow_html=True)

# Formulaire de saisie des données
with st.form("Formulaire de saisie", clear_on_submit=True):
    st.markdown("## Formulaire")
    age = st.number_input("Age")
    anciennete = st.number_input("Anciennete")
    notation_pret = st.selectbox("notation_pret", options=["A", "B", "C", "D", "E"], index=0)
    montant_pret = st.number_input("Montant Pret")
    pourcentage_pret_revenu = st.number_input("Pourcentage Pret Revenu")
    taux_interet = st.number_input("Taux d'interet")
    propriete = st.selectbox("propriete", options=["Locataire", "Proprietaire", "Hypotheque", "Autres"], index=0)
    motif_pret = st.selectbox("motif_pret", options=["Personnel", "Education", "Entreprendre", "Medical", "Travaux_maison", "Regroupement_dettes"], index=0)
    historique_defaut = st.selectbox("historique_defaut", options=["N", "Y"], index=0)
    revenu = st.number_input("Revenue")
    duree_historique_credit = st.number_input("Duree historique credit")

    valide = st.form_submit_button('Valider')

    if valide:
        features = [age, anciennete, notation_pret, montant_pret, pourcentage_pret_revenu, taux_interet, propriete, motif_pret, historique_defaut, revenu, duree_historique_credit]
        input_data = pd.DataFrame(features).transpose()
        input_data.columns = ["age", "anciennete", "notation_pret", "montant_pret", "pourcentage_pret_revenu", "taux_interet", "propriete", "motif_pret", "historique_defaut", "revenu", "duree_historique_credit"]

        result = model.predict(input_data)
        proba = model.predict_proba(input_data)[:, 1]

        st.session_state['prediction_score'] = result[0]
        st.session_state['prediction_status'] = "Approuvé" if result[0] else "Refusé"
        st.session_state['prediction_model'] = "Régression Logistique"
        st.session_state['prediction_proba'] = proba

# Afficher les résultats à droite
if st.session_state['prediction_score'] is not None:
    st.markdown("<div class='stResults'>", unsafe_allow_html=True)
    st.write("Résultats")
    st.write(f"Score: {st.session_state['prediction_proba']}")
    st.write(f"Status: {st.session_state['prediction_status']}")
    st.write(f"Modèle: {st.session_state['prediction_model']}")
    commentaire = st.text_input("Commentaire")
    validate = st.button("Envoyer")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
page_bg_css = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color:rgb(0, 83, 139);
        color: white; /* Pour rendre le texte lisible sur un fond sombre */
         font-weight: bold;
    }
    [data-testid="stForm"] label, [data-testid="stForm"] input, [data-testid="stForm"] textarea {
    color: black; /* Mettre le texte des formulaires en noir */
     font-weight: bold;
    }

    </style>
    """
st.markdown(page_bg_css, unsafe_allow_html=True)