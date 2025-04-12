import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier
import os
from datetime import datetime, timedelta

# Initialisation du fichier historique avec date
history_file = "dataset/weather_history.csv"
if not os.path.exists(history_file):
    pd.DataFrame(columns=["prediction", "date"]).to_csv(history_file, index=False)

# Titre de l'application
st.title("🌦️ Prédiction Météorologique Avancée")
st.markdown("Cette application prédit le temps qu'il fera avec des statistiques détaillées.")

# Charger le modèle
try:
    model = joblib.load("model_xgboost.pkl")
except:
    st.error("⚠️ Erreur : Modèle non trouvé. Vérifiez le chemin.")
    st.stop()

# Entrées utilisateur
st.sidebar.header("🔧 Paramètres Météo")
precipitation = st.sidebar.slider("Précipitation (mm)", 0.0, 20.0, 2.5)
temp_max = st.sidebar.slider("Température Max (°C)", -10.0, 50.0, 25.0)
temp_min = st.sidebar.slider("Température Min (°C)", -10.0, 50.0, 10.0)
wind = st.sidebar.slider("Vitesse du Vent (km/h)", 0.0, 100.0, 15.0)

# Bouton de prédiction
if st.sidebar.button("Prédire le Temps"):
    input_data = pd.DataFrame([[precipitation, temp_max, temp_min, wind]], 
                            columns=["precipitation", "temp_max", "temp_min", "wind"])
    
    prediction = model.predict(input_data)[0]
    weather_types = {0: "Bruine", 1: "Brouillard", 2: "Pluie", 3: "Neige", 4: "Soleil"}
    weather_icons = {0: "☔", 1: "🌫️", 2: "🌧️", 3: "❄️", 4: "☀️"}
    
    # Ajouter à l'historique avec date
    df_history = pd.read_csv(history_file)
    new_entry = pd.DataFrame([{
        "prediction": prediction,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "precipitation": precipitation,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "wind": wind
    }])
    
    df_history = pd.concat([df_history, new_entry], ignore_index=True)
    df_history.to_csv(history_file, index=False)
    
    st.success(f"### Résultat : {weather_icons[prediction]} {weather_types[prediction]}")
    st.markdown("---")

# Affichage des statistiques avancées
st.subheader("📊 Statistiques Météo Détaillées")

if os.path.exists(history_file):
    df_stats = pd.read_csv(history_file)
    
    # Vérification si la colonne 'date' existe
    if 'date' in df_stats.columns:
        df_stats['date'] = pd.to_datetime(df_stats['date'])
    else:
        st.error("La colonne 'date' est manquante dans les données historiques.")
        st.stop()
    
    if not df_stats.empty:
        # Filtre des 2 derniers mois
        two_months_ago = datetime.now() - timedelta(days=60)
        df_recent = df_stats[df_stats['date'] >= two_months_ago]
        
        # Dictionnaire des types de temps
        labels = {0: "Bruine", 1: "Brouillard", 2: "Pluie", 3: "Neige", 4: "Soleil"}
        icons = {0: "☔", 1: "🌫️", 2: "🌧️", 3: "❄️", 4: "☀️"}
        
        # 1. Statistiques générales
        st.markdown("### 🌡️ Données Globales (2 mois)")
        
        # Utilisation de st.columns sans assignation à des variables
        cols = st.columns(3)
        cols[0].metric("Précipitations moyennes", f"{df_recent['precipitation'].mean():.1f} mm")
        cols[1].metric("Température max moyenne", f"{df_recent['temp_max'].mean():.1f}°C")
        cols[2].metric("Température min moyenne", f"{df_recent['temp_min'].mean():.1f}°C")
        
        # 2. Dernières précipitations (7 dernières)
        st.markdown("### 🌧️ 7 Dernières Précipitations")
        last_rain = df_stats[df_stats['prediction'].isin([0,2,3])].tail(7)  # Bruine, Pluie, Neige
        
        if not last_rain.empty:
            for idx, row in last_rain.iterrows():
                if pd.isna(row['date']):
                    st.write("Date manquante")
                else:
                    st.write(f"{row['date'].strftime('%d/%m %H:%M')} - {icons[row['prediction']]} {labels[row['prediction']]} - {row['precipitation']} mm")
        else:
            st.write("Aucune précipitation récente enregistrée.")
        
        # 3. Histogrammes
        st.markdown("### 📈 Répartition des Types de Temps")
        
        # Préparation des données
        df_stats['weather_type'] = df_stats['prediction'].map(labels)
        df_stats['weather_icon'] = df_stats['prediction'].map(icons)
        
        # Histogramme général
        st.bar_chart(df_stats['weather_type'].value_counts())
        
        
        
        # 4. Tableau complet des données
        st.markdown("### 📝 Données Complètes")
        st.dataframe(df_stats.sort_values('date', ascending=False).head(20))
        
    else:
        st.write("Aucune donnée historique disponible.")
else:
    st.write("Aucune donnée historique disponible.")