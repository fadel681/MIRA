from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib

# Charger les données et entraîner le modèle
def train_and_save_model():
    # Charger les données à partir d'un fichier CSV
    try:
        data = pd.read_csv("C:\\Users\\fdiakhat\\OneDrive - Capgemini\\Desktop\\MIRA\\Test1\\clean_RiskCredit.csv", sep=';')
    except FileNotFoundError:
        print("Le fichier spécifié est introuvable. Veuillez vérifier le chemin du fichier.")
        return

    # Supprimer les valeurs manquantes
    data.dropna(inplace=True)

    # Définir les colonnes
    numeric_features = ['age', 'revenu', 'anciennete', 'montant_pret', 'taux_interet', 'pourcentage_pret_revenu', 'duree_historique_credit']
    categorical_features = ['propriete', 'motif_pret', 'notation_pret', 'historique_defaut']

    # Transformer numérique : MinMaxScaler
    numeric_transformer = MinMaxScaler()

    # Transformer catégorien : OneHotEncoder (équivalent de get_dummies)
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocesseur = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)]
    )

    # Pipeline complète avec la régression logistique
    model = Pipeline(steps=[("preprocessor", preprocesseur), ("classifier", LogisticRegression())])

    # Diviser les données en ensemble d'apprentissage et de test
    X = data[numeric_features + categorical_features]  # Caractéristiques
    y = data['statut_pret']  # Étiquette
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle sur l'ensemble d'apprentissage
    model.fit(X_train, y_train)

    # Sauvegarder le modèle
    joblib.dump(model, 'votre_modele.pkl')

# Charger le modèle
def load_model():
    model = joblib.load('votre_modele.pkl')  # Charger le modèle sauvegardé
    return model

# Fonction de prédiction
def predict(model, input_data):
    prediction = model.predict([input_data])  # Faire une prédiction avec le modèle
    return prediction

# Entraîner et sauvegarder le modèle (à exécuter une seule fois)
train_and_save_model()