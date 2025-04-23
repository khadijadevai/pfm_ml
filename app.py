import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import datetime
import os
import base64
import re
import urllib.parse
import gdown
# Chargement du modèle

# ID du fichier sur Google Drive
url = 'https://drive.google.com/file/d/1NmJ4BUykN0nlJtLQnp_iqfTRapDNNMtw/view?usp=drive_link'
output = 'model2.cbm'

# Vérifie si le modèle existe déjà en local
if not os.path.exists(output):
    with st.spinner('Téléchargement du modèle...'):
        try:
            gdown.download(url, output, quiet=False)
        except Exception as e:
            st.error(f"Échec du téléchargement du modèle : {e}")
            st.stop()

# Chargement du modèle
model = CatBoostRegressor()
try:
    model.load_model(output)
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# Données de référence
df_ref = pd.read_csv("data_final.csv")
cat_features = ['Boite', 'Carburant', 'secteur', 'marque', 'model', 'origine', 'premiére main', 'état', 'km_binned']

# Chemin de l'historique
HISTORIQUE_PATH = "historique_predictions.csv"
if not os.path.exists(HISTORIQUE_PATH):
    pd.DataFrame(columns=list(df_ref.drop(columns=["Prix"]).columns) + ["prediction", "timestamp"]).to_csv(HISTORIQUE_PATH, index=False)

# Style & fond

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_base64 = get_base64_image("bg.jpg")

st.set_page_config(page_title="Prédicteur Avito", layout="wide")

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
.main {{
    background-color: rgba(0, 0, 0, 0.65);
    padding: 2rem;
    border-radius: 15px;
    color: white;
}}
.title {{
    color: #9f2929;
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
}}
.stTabs [role="tab"] {{
    background-color: #9f2929;
    padding: 10px;
    color: white;
    border-radius: 10px 10px 0 0;
}}
.stTabs [role="tab"][aria-selected="true"] {{
    background-color: #ffffff;
    color: #9f2929;
}}
.stButton>button, .stDownloadButton>button {{
    background-color: #9f2929;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    padding: 10px 24px;
    border: none;
}}
.st-emotion-cache-h4xjwg{{ 

    display: none;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Prédiction du Prix des Voitures d\'Occasion</div>', unsafe_allow_html=True)

# Fonctionnalités préservées : fonctions existantes de prétraitement, prédiction, etc.
# ... (les fonctions comme preprocess_input, faire_prediction, construire_url_avito sont inchangées)

def extraire_km_moyen(km_str):
    km_str = str(km_str) 
    chiffres = re.findall(r'\d+', km_str.replace(" ", ""))
    if len(chiffres) == 2:
        km_min = int(chiffres[0])
        km_max = int(chiffres[1])
        return (km_min + km_max) // 2
    elif len(chiffres) == 1:
        return int(chiffres[0])
    else:
        return 0  # Valeur par défaut si format inconnu

# Fonction pour binner le kilométrage
def get_km_binned(km):
    if km < 50000:
        return "0-50k"
    elif km < 100000:
        return "50k-100k"
    elif km < 150000:
        return "100k-150k"
    elif km < 200000:
        return "150k-200k"
    elif km < 300000:
        return "200k-300k"
    elif km < 400000:
        return "300k-400k"
    elif km < 500000:
        return "400k-500k"
    else:
        return "500k+"

def preprocess_input(donnees):
    df = pd.DataFrame([donnees])
    df["kilometrage_numerique"] = df["kilometrage"].apply(extraire_km_moyen)
    df["km_moyen"] = df["kilometrage_numerique"]
    df["km_binned"] = df["kilometrage_numerique"].apply(get_km_binned)
    
    df["premiére main"] = df["premiére main"].astype(str)
    df["model"] = df["model"].astype(str)

    colonnes_utiles = [col for col in df_ref.columns if col not in ['Prix', 'kilometrage', 'km_max', 'km_min']]
    return df[colonnes_utiles]
# Fonction de prédiction
def faire_prediction(donnees):
    df = preprocess_input(donnees)
    pool = Pool(df, cat_features=cat_features)
    return model.predict(pool)[0]

import urllib.parse

def construire_url_avito(entree):
    base_url = "https://www.avito.ma/fr/casablanca/voitures_d_occasion-%C3%A0_vendre"
    marques_dict = {
        "mercedes-benz": 41, "renault": 49, "peugeot": 46, "toyota": 56, "ford": 18,
        "volkswagen": 58, "fiat": 17, "hyundai": 24, "nissan": 44, "chevrolet": 10,
        "kia": 30, "citroen": 12, "mazda": 47, "opel": 45, "suzuki": 55, "bmw": 5,
        "honda": 22, "audi": 3, "alfa romeo": 1, "seat": 50, "bentley": 4, "lada": 72,
        "cadillac": 7, "volvo": 59, "dfsk": 67, "jeep": 29, "chery": 9, "porsche": 48,
        "skoda": 51, "daewoo": 61, "jaguar": 28, "rover": 62, "ssangyong": 53, "geely": 20,
        "mitsubishi": 43, "land rover": 33, "lexus": 34, "mini": 42, "ds": 66, "dacia": 13,
        "ferrari": 16, "maserati": 38, "mg": 97, "aston martin": 2, "acura": 65, "chrysler": 11,
        "daihatsu": 14, "infiniti": 25, "gwm motors": 75, "lamborghini": 31, "lancia": 32,
        "mahindra": 36, "tesla": 68, "ac": 63, "alpine": 98, "changan": 80, "isuzu": 26,
        "smart": 52, "subaru": 54, "abarth": 73, "byd": 6, "cupra": 71, "dodge": 15,
        "gaz": 74, "gmc": 21, "hummer": 23, "iveco": 27, "leapmotor": 112, "pontiac": 47,
        "tata": 77, "acrea": 64, "autres": 120, "changhe": 8, "faw": 99, "force": 69,
        "foton": 19, "lifan": 76, "lincoln": 35, "rolls-royce": 70, "seres": 111,
        "ufo": 57, "zotye": 60
    }

    def convertir_kilometrage(km):
        if km < 0:
            raise ValueError("Le kilométrage ne peut pas être négatif")
        elif km < 5000:
            return 0
        elif km < 10000:
            return 1
        elif km < 15000:
            return 2
        elif km < 20000:
            return 3
        elif km < 25000:
            return 4
        elif km < 30000:
            return 5
        elif km < 35000:
            return 6
        elif km < 40000:
            return 7
        elif km < 45000:
            return 8
        elif km < 50000:
            return 9
        elif km < 55000:
            return 10
        elif km < 60000:
            return 11
        elif km < 65000:
            return 12
        elif km < 70000:
            return 13
        elif km < 75000:
            return 14
        elif km < 80000:
            return 15
        elif km < 85000:
            return 16
        elif km < 90000:
            return 17
        elif km < 95000:
            return 18
        elif km < 100000:
            return 19
        elif km < 105000:
            return 20
        elif km < 110000:
            return 21
        elif km < 115000:
            return 22
        elif km < 120000:
            return 23
        elif km < 125000:
            return 24
        elif km < 130000:
            return 25
        elif km < 135000:
            return 26
        elif km < 140000:
            return 27
        elif km < 145000:
            return 28
        elif km < 150000:
            return 29
        elif km < 200000:
            return 30
        elif km < 250000:
            return 31
        elif km < 300000:
            return 32
        elif km < 350000:
            return 33
        elif km < 400000:
            return 34
        elif km < 450000:
            return 35
        elif km < 500000:
            return 36
        elif km < 550000:
            return 37
        elif km < 600000:
            return 38
        elif km < 650000:
            return 39
        elif km < 700000:
            return 40
        else:
            return 41  # pour 700000 km et plus


    params = {
        "brand": marques_dict.get(entree["marque"].lower(), -1),
        "model": str(entree["model"]).replace(" ", "").lower(),
        "fuel": 1 if entree["Carburant"] == "Diesel" else 2 if entree["Carburant"] == "Essence" else 3 if entree["Carburant"] == "Électrique" else 4 if entree["Carburant"] == "LPG" else 5,
        "regdate": f"{entree['Année']}-{entree['Année'] + 2}",
        "mileage": convertir_kilometrage(entree["kilometrage"]),
        # "pfiscale": f"{entree['puissance fiscale']-1}-{entree['puissance fiscale']+1}",
        "bv": 0 if str.lower(entree["Boite"]) == "automatique" else 1,
        "first_owner": 0 if entree["premiére main"] == "Oui" else 1,
        "doors": 1 if int(entree["nbr de portes"])==5 else 0,
        "v_origin": 0 if str.lower(entree["origine"]) == "dédouanée" else 1 if str.lower(entree["origine"]) == "pas encore dédouanée" else 2 if str.lower(entree["origine"]) == "ww au maroc" else 3,
        "auto_condition": 0 if str.lower(entree["état"]) == "excellent" else 1 if str.lower(entree["état"]) == "très bon" else 2 if str.lower(entree["état"]) == "bon" else 3 if str.lower(entree["état"]) == "correct" else 4 if str.lower(entree["état"]) == "endommagé" else 5 if str.lower(entree["état"]) == "pour pièces" else 6 if str.lower(entree["état"]) == "neuf" else 7
    }

    # Filtrer les None
    params = {k: v for k, v in params.items() if v is not None}

    # Encoder proprement l'URL
    query_string = urllib.parse.urlencode(params)

    return f"{base_url}?{query_string}"
# Interface onglets
onglet = st.tabs(["Prédiction", "Annonces Similaires", "Historique"])

with onglet[0]:
    st.markdown("**Remplis le formulaire ci-dessous pour obtenir une estimation basée sur les données Avito.**")
    with st.form("formulaire"):
        col1, col2, col3 = st.columns(3)

        with col1:
            marque = st.selectbox("Marque", sorted(df_ref["marque"].unique()))
            modele = st.text_input("Modèle", "Logan")
            secteur = st.selectbox("Secteur", sorted(df_ref["secteur"].unique()))
            annee = st.number_input("Année", min_value=1990, max_value=datetime.datetime.now().year, value=2018)

        with col2:
            kilometrage = st.number_input("Kilométrage", min_value=0, max_value=500000, value=100000)
            puissance_fiscale = st.number_input("Puissance Fiscale", min_value=2, max_value=50, value=6)
            nb_portes = st.selectbox("Nombre de portes", sorted(df_ref["nbr de portes"].unique()))
            premiere_main = st.selectbox("Première main", ["Oui", "Non"])

        with col3:
            origine = st.selectbox("Origine", sorted(df_ref["origine"].unique()))
            etat = st.selectbox("État", sorted(df_ref["état"].unique()))
            boite = st.selectbox("Boîte de vitesses", sorted(df_ref["Boite"].unique()))
            carburant = st.selectbox("Carburant", sorted(df_ref["Carburant"].unique()))

        submitted = st.form_submit_button("🎯 Prédire le prix")

        if submitted:
            entree = {
                "marque": marque,
                "model": modele,
                "secteur": secteur,
                "kilometrage": kilometrage,
                "nbr de portes": nb_portes,
                "puissance fiscale": puissance_fiscale,
                "origine": origine,
                "état": etat,
                "premiére main": premiere_main,
                "Année": annee,
                "Boite": boite,
                "Carburant": carburant,
            }
            prediction = faire_prediction(entree)
            st.success(f"💰 **Prix estimé : {format(int(prediction), ',').replace(',', ' ')} DH**")

            donnees_finales = preprocess_input(entree)
            donnees_finales["prediction"] = int(prediction)
            donnees_finales["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            historique_df = pd.read_csv(HISTORIQUE_PATH)
            historique_df = pd.concat([historique_df, donnees_finales], ignore_index=True)
            historique_df.to_csv(HISTORIQUE_PATH, index=False)

with onglet[1]:
    st.markdown("### 📍 Annonces Similaires sur Avito")
    if 'entree' in locals():
        url_avito = construire_url_avito(entree)
        st.components.v1.iframe(url_avito, height=600, scrolling=True)
    else:
        st.info("Veuillez d'abord effectuer une prédiction pour voir les annonces similaires.")

with onglet[2]:
    st.markdown("### 📄 Historique des Prédictions")
    historique_df = pd.read_csv(HISTORIQUE_PATH)
    st.dataframe(historique_df.sort_values(by="timestamp", ascending=False).head(20), use_container_width=True)

    csv_export = historique_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📅 Télécharger le rapport complet (CSV)",
        data=csv_export,
        file_name="rapport_predictions.csv",
        mime="text/csv"
    )
