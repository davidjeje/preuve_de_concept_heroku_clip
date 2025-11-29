# # Base Python légère
# FROM python:3.10-slim

# # Installer dépendances système nécessaires pour PIL / PyTorch
# RUN apt-get update && apt-get install -y \
#     git \
#     libgl1 \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# # Créer le répertoire de l'app
# WORKDIR /app

# # Copier les fichiers de ton projet
# COPY . /app

# # Installer pip et dépendances Python
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# # Exposer le port Streamlit
# EXPOSE 8501

# # Lancer Streamlit
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ==============================
# Dockerfile pour Heroku + Streamlit + S3
# ==============================

# Image de base Python légère
# Base Python légère
# FROM python:3.11-slim

# # Installer dépendances système nécessaires pour PIL / PyTorch + git
# RUN apt-get update && apt-get install -y \
#     git \
#     libgl1 \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# # Créer le répertoire de l'app
# WORKDIR /app

# # Copier uniquement les fichiers nécessaires
# COPY requirements.txt .streamlit app.py ./

# # Installer pip et dépendances Python
# RUN pip install --no-cache-dir --upgrade pip \
#     && pip install --no-cache-dir -r requirements.txt

# # Exposer le port Streamlit
# EXPOSE 8501

# # Lancer Streamlit
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.11-slim
ENV PORT=8501
RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY app.py .
RUN mkdir -p models data
EXPOSE $PORT
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]