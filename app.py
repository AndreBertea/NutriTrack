import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Création du dossier "data" s'il n'existe pas
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)


st.sidebar.title("Ajouter un rapport nutritionnel")
st.sidebar.text("ChatGPT ")
markdown_input = st.sidebar.text_area("Collez votre texte Markdown ici", height=300)

# Sauvegarder le texte collé sous forme de fichier Markdown avec la date du jour
if st.sidebar.button("Sauvegarder"):
    if markdown_input:
        today = datetime.now().strftime("%y-%m-%d")
        file_path = os.path.join(data_folder, f"{today}.md")
        with open(file_path, "w") as file:
            file.write(markdown_input)
        st.sidebar.success(f"Fichier sauvegardé sous : {today}.md")
    else:
        st.sidebar.error("Veuillez coller du texte Markdown avant de sauvegarder.")

# Récupérer la liste des fichiers Markdown disponibles dans le dossier "data"
files = [f for f in os.listdir(data_folder) if f.endswith('.md')]
selected_file = st.selectbox("Sélectionnez un rapport disponible", files)

# Affichage du contenu du fichier sélectionné
if selected_file:
    file_path = os.path.join(data_folder, selected_file)
    with open(file_path, "r") as file:
        markdown_content = file.read()
        st.markdown(markdown_content)
        
        # Simulation des données extraites du fichier Markdown (à adapter selon ton contenu réel)
        # Exemples de données fictives pour affichage
        data = {
            'Nutrient': ['Calories', 'Protein', 'Carbs', 'Fat', 'Calcium', 'Vitamin D', 'Magnesium', 'Vitamin B12', 'Vitamin K2'],
            'Intake': [1500, 70, 200, 60, 800, 400, 250, 5, 80],  # Remplace par les données réelles extraites
            'RDI': [2500, 70, 300, 70, 1000, 600, 400, 2.4, 120]
        }
        
        df = pd.DataFrame(data)
        df['Percentage'] = (df['Intake'] / df['RDI']) * 100
        
        # Affichage des analyses nutritionnelles
        st.title("Daily Nutritional Analysis")
        for index, row in df.iterrows():
            color = 'green' if row['Percentage'] >= 100 else 'orange' if row['Percentage'] >= 70 else 'red'
            st.markdown(f"**{row['Nutrient']}**: {row['Intake']} ({row['Percentage']:.1f}%)", unsafe_allow_html=True)
            # Graphique en barre pour les apports
            plt.bar(row['Nutrient'], row['Percentage'], color=color)
        
        st.pyplot(plt)
