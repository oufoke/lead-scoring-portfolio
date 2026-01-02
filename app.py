import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Lead Scoring IA", layout="wide")

# --- CHARGEMENT DU CERVEAU (CACHE) ---
# On utilise le cache pour ne pas recharger le modèle à chaque clic
@st.cache_resource
def load_model_data():
    artifacts = joblib.load('model_data.pkl')
    return artifacts['model'], artifacts['features']

model, feature_names = load_model_data()

# --- INTERFACE : BARRE LATÉRALE (INPUTS) ---
st.sidebar.header("👤 Profil du Prospect")

# On recrée les champs de saisie comme dans le fichier Excel
secteur = st.sidebar.selectbox("Secteur d'activité", ['Tech', 'Sante', 'Retail', 'Industrie', 'Education'])
taille = st.sidebar.selectbox("Taille Entreprise", ['1-10', '11-50', '51-200', '200+'])
poste = st.sidebar.selectbox("Poste du Contact", ['Stagiaire', 'Operationnel', 'Manager', 'Directeur', 'VP/C-Level'])
source = st.sidebar.selectbox("Source du Lead", ['Cold Call', 'Emailing', 'Webinar', 'Site Web', 'Linkedin'])

st.sidebar.markdown("---")
st.sidebar.header("📊 Comportement")
temps_site = st.sidebar.slider("Temps sur site (secondes)", 0, 600, 180)
emails = st.sidebar.slider("Emails ouverts", 0, 10, 3)
pages_vues = st.sidebar.slider("Pages Vues", 0, 20, 4)
recence = st.sidebar.number_input("Dernière activité (Jours)", 0, 365, 5)

# --- MOTEUR DE PRÉDICTION ---
# 1. On crée un DataFrame avec les données saisies
input_data = pd.DataFrame({
    'Taille_Entreprise': [taille],
    'Poste_Contact': [poste],
    'Secteur': [secteur],
    'Source_Lead': [source],
    'Temps_Sur_Site_Sec': [temps_site],
    'Pages_Vues': [pages_vues],
    'Emails_Ouverts': [emails],
    'Derniere_Activite_Jours': [recence]
})

# 2. On transforme les données comme lors de l'entraînement (One-Hot Encoding)
input_encoded = pd.get_dummies(input_data)

# 3. ALIGNEMENT DES COLONNES (CRUCIAL)
# Si une catégorie n'est pas sélectionnée (ex: pas de "Sante"), la colonne n'existe pas.
# On doit forcer la structure pour qu'elle soit identique à celle de l'entraînement.
input_final = input_encoded.reindex(columns=feature_names, fill_value=0)

# 4. Prédiction
prediction_proba = model.predict_proba(input_final)[0][1] # Probabilité de classe 1 (Signé)
score = round(prediction_proba * 100, 1)

# --- INTERFACE : ZONE PRINCIPALE ---

st.title("🚀 Assistant Intelligent de Vente B2B")
st.markdown("Cet outil aide les équipes commerciales à **prioriser les appels** grâce à l'IA.")

# Colonnes pour l'affichage du Score
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Probabilité de Signature")
    # Changement de couleur selon le score
    color = "red"
    if score > 40: color = "orange"
    if score > 70: color = "green"
    
    # MODIFICATION ICI : On ajoute :.1f pour forcer 1 chiffre après la virgule
    st.markdown(f"<h1 style='color:{color}; font-size: 70px;'>{score:.1f}/100</h1>", unsafe_allow_html=True)
    
    if score > 70:
        st.success("🔥 LEAD TRÈS CHAUD : À appeler en priorité !")
    elif score > 40:
        st.warning("⚠️ LEAD TIÈDE : À nourrir (Nurturing).")
    else:
        st.error("❄️ LEAD FROID : Peu de chance.")

with col2:
    st.subheader("🧠 Pourquoi ce score ? (IA Explicable)")
    st.info("Le graphique ci-dessous montre quels critères ont fait monter (Rouge) ou descendre (Bleu) la note.")
    
    # Calcul des valeurs SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_final)
    
    # Création du graphique Waterfall
    fig, ax = plt.subplots(figsize=(8, 4))
    # Note: shap.plots.waterfall attend un objet Explanation, on ruse un peu pour l'affichage simple
    shap.plots.waterfall(shap.Explanation(values=shap_values[0], 
                                          base_values=explainer.expected_value, 
                                          data=input_final.iloc[0], 
                                          feature_names=feature_names), 
                         max_display=8, show=False)
    st.pyplot(fig)

# --- FOOTER ---
st.markdown("---")
st.caption("Prototype Portfolio - Data Product Management - Oumar DPM")