import streamlit as st
import pandas as pd
from model_utils import load_model, predict
from PIL import Image
import base64
import os

# Charger le modèle
model = load_model()

vars_sessions = ["prediction_score", "prediction_status", "prediction_model", "prediction_proba"]
for v in vars_sessions:
    if v not in st.session_state:
        st.session_state[v] = None

# Charger le logo et l'image d'arrière-plan
logo_path = os.path.join('media', 'capgemini-symbol.png')
background_path = os.path.join('media', 'capgemini_streamlit.jfif')

logo = Image.open(logo_path)
background = background_path

# Convertir l'image d'arrière-plan en base64
with open(background, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Afficher le logo réduit à gauche
st.sidebar.image(logo, use_container_width=True, width=100)

# Appliquer l'image d'arrière-plan à toute la page
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #ffffff; /* Blanc pour le texte */
        font-family: 'Arial', sans-serif; /* Police plus moderne */
    }}
    .stSidebar {{
        background-color: #ffffff; /* Blanc */
    }}
    .stHeader {{
        color: #000000; /* Noir pour le texte de l'en-tête */
        font-size: 2em; /* Taille de police pour l'en-tête */
        font-weight: bold; /* En-tête en gras */
        text-align: center; /* Centrer l'en-tête */
        margin-top: 20px; /* Marge supérieure pour espacer l'en-tête */
        margin-bottom: 20px; /* Marge inférieure pour espacer l'en-tête */
    }}
    .stFormContainer {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh; /* Hauteur de la fenêtre */
    }}
    .stForm {{
        background-color: rgba(51, 51, 51, 0.8); /* Fond noir clair avec transparence pour le formulaire */
        padding: 20px; /* Espacement intérieur pour le formulaire */
        border-radius: 10px; /* Coins arrondis pour le formulaire */
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5); /* Ombre pour le formulaire */
        color: #ffffff; /* Blanc pour le texte du formulaire */
    }}
    .stForm input, .stForm label {{
        color: #ffffff; /* Blanc pour les entrées et les labels */
    }}
    .stForm button {{
        background-color: #000000; /* Noir pour le bouton */
        color: #ffffff; /* Blanc pour le texte du bouton */
        border: none; /* Pas de bordure */
        padding: 10px 20px; /* Espacement intérieur pour le bouton */
        border-radius: 5px; /* Coins arrondis pour le bouton */
        cursor: pointer; /* Curseur en forme de main */
    }}
    .stForm button:hover {{
        background-color: #333333; /* Noir clair pour le bouton au survol */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# En-tête de l'application
st.markdown("<div class='stHeader'>Application de prédiction de crédit</div>", unsafe_allow_html=True)
st.markdown("<div class='stHeader'>Bienvenue sur notre application de prédiction de crédit !</div>", unsafe_allow_html=True)

# Formulaire de saisie des données
st.markdown("<div class='stFormContainer'>", unsafe_allow_html=True)
with st.form("Formulaire de saisie"):
    st.write("Formulaire")
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
        st.session_state['prediction_status'] = "Approuvé" si result[0] == 1 sinon "Refusé"
        st.session_state['prediction_model'] = "Régression Logistique"
        st.session_state['prediction_proba'] = proba
st.markdown("</div>", unsafe_allow_html=True)

# Formulaire de réponse
if st.session_state['prediction_score'] is not None:
    st.markdown("<div class='stFormContainer'>", unsafe_allow_html=True)
    with st.form("Formulaire de reponse"):
        st.write("Resultat")
        cols = st.columns(3)
        score = cols[0].text(f"score: {st.session_state['prediction_proba']}")
        status = cols[1].text(f"status: {st.session_state['prediction_status']}")
        modele = cols[2].text(st.session_state['prediction_model'])
        commentaire = st.text_input("Commentaire")
        validate = st.form_submit_button("Envoyer")
    st.markdown("</div>", unsafe_allow_html=True)