# Utiliser une image Python de base
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY . /app

# Mettre à jour pip et installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y libopenblas-dev

# Mettre à jour pip avant d'installer les dépendances Python
RUN pip install --upgrade pip

# Installer les dépendances Python depuis requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par Streamlit
EXPOSE 8501

# Commande pour lancer l'application Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
