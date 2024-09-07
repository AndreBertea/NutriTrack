# Utiliser une image Python de base
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY . /app

# Mettre à jour et installer les dépendances système
RUN apt-get update && apt-get install -y libopenblas-dev pkg-config libhdf5-dev

# Mettre à jour pip avant d'installer les dépendances Python
RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y gcc g++ libhdf5-dev

# Installer les dépendances Python depuis requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par Streamlit
EXPOSE 8501

# Commande pour lancer l'application Streamlit
CMD ["streamlit", "run", "bourse.py", "--server.port=8501", "--server.headless=true"]
