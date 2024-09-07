FROM python:3.9-slim

WORKDIR /app

COPY . /app

# Mettre à jour pip avant d'installer les dépendances
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "bourse.py", "--server.port=8501", "--server.headless=true"]
