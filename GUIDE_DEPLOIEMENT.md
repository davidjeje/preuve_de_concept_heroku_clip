# 🚀 Guide Complet de Déploiement - Application Streamlit CLIP sur Heroku

## 📦 Fichiers fournis

Voici tous les fichiers que je vous ai créés :

1. **app.py** - Application Streamlit principale
2. **requirements.txt** - Dépendances Python
3. **Procfile** - Configuration Heroku
4. **setup.sh** - Configuration Streamlit
5. **.gitignore** - Fichiers à ignorer
6. **download_models.py** - Script téléchargement modèles MLflow
7. **README.md** - Documentation
8. **GUIDE_DEPLOIEMENT.md** - Ce guide (instructions détaillées)

---

## ✅ Checklist complète

### Phase 1 : Préparation locale

- [ ] Créer dossier projet
- [ ] Copier tous les fichiers fournis
- [ ] Télécharger modèles depuis MLflow
- [ ] Préparer dataset (optionnel)
- [ ] Tester localement

### Phase 2 : Configuration Git

- [ ] Initialiser Git
- [ ] Créer .gitignore
- [ ] Premier commit

### Phase 3 : Déploiement Heroku

- [ ] Créer compte Heroku
- [ ] Installer Heroku CLI
- [ ] Créer app Heroku
- [ ] Déployer
- [ ] Tester en ligne

---

## 📋 Instructions détaillées étape par étape

### ÉTAPE 1 : Créer la structure du projet

```bash
# Créer dossier projet
mkdir clip-heroku-app
cd clip-heroku-app

# Créer sous-dossiers
mkdir models
mkdir data
```

### ÉTAPE 2 : Copier les fichiers fournis

Copiez tous les fichiers des artifacts dans votre dossier :

```
clip-heroku-app/
├── app.py
├── requirements.txt
├── Procfile
├── setup.sh
├── .gitignore
├── download_models.py
├── README.md
└── GUIDE_DEPLOIEMENT.md
```

### ÉTAPE 3 : Télécharger les modèles depuis MLflow

**Option A : Script automatique (recommandé)**

```bash
# Assurez-vous que MLflow est lancé
# Dans un autre terminal : poetry run mlflow ui

# Exécuter le script
python download_models.py
```

Le script va :
1. Lister tous vos runs MLflow
2. Vous demander le Run ID à télécharger
3. Télécharger automatiquement les 2 fichiers dans `models/`

**Option B : Téléchargement manuel**

Dans un notebook Jupyter :

```python
import mlflow
import shutil
import os

# Configuration
mlflow.set_tracking_uri("http://localhost:5000")
run_id = "VOTRE_RUN_ID"  # Copiez depuis MLflow UI

# Créer dossier
os.makedirs("models", exist_ok=True)

# Télécharger
client = mlflow.tracking.MlflowClient()

# LightGBM
artifacts = client.list_artifacts(run_id)
for art in artifacts:
    if "lgbm" in art.path.lower():
        client.download_artifacts(run_id, art.path, "models")
        
# Label encoder
for art in artifacts:
    if "label_encoder" in art.path.lower():
        client.download_artifacts(run_id, art.path, "models")

print("✅ Modèles téléchargés!")
```

**Vérification :**

```bash
ls -lh models/

# Vous devriez voir :
# lgbm_clip_classifier.pkl  (~5-10 MB)
# label_encoder.pkl         (~1-5 KB)
```

### ÉTAPE 4 : Préparer le dataset (optionnel)

Si vous voulez le mode "Test sur dataset" :

```python
# Dans votre notebook
df_clean.to_csv("data/df_clean.csv", index=False)
```

⚠️ **IMPORTANT** : 
- Ne mettez PAS le dossier `data/Images/` (trop volumineux)
- Seulement `df_clean.csv` avec colonnes : `text_clip`, `image_path`, `category`
- Heroku a une limite de 500MB pour le slug

### ÉTAPE 5 : Tester localement

```bash
# Installer dépendances
pip install -r requirements.txt

# Lancer l'app
streamlit run app.py
```

Ouvrez http://localhost:8501 et testez :
1. Mode "Upload Image + Texte"
2. Si dataset présent : Mode "Tester sur Dataset"

### ÉTAPE 6 : Initialiser Git

```bash
# Initialiser Git
git init

# Ajouter tous les fichiers
git add .

# Premier commit
git commit -m "Initial commit: CLIP Streamlit app"
```

### ÉTAPE 7 : Créer compte Heroku

1. Aller sur https://signup.heroku.com/
2. Créer un compte gratuit
3. Vérifier email

### ÉTAPE 8 : Installer Heroku CLI

**Windows** :
- Télécharger depuis https://devcenter.heroku.com/articles/heroku-cli
- Installer et redémarrer terminal

**Mac** :
```bash
brew tap heroku/brew && brew install heroku
```

**Linux** :
```bash
curl https://cli-assets.heroku.com/install.sh | sh
```

**Vérifier installation** :
```bash
heroku --version
```

### ÉTAPE 9 : Se connecter à Heroku

```bash
heroku login
# Appuyer sur une touche, navigateur s'ouvre
# Se connecter dans le navigateur
```

### ÉTAPE 10 : Créer app Heroku

```bash
# Créer une nouvelle app (nom doit être unique)
heroku create votre-app-clip-demo

# OU laisser Heroku générer un nom aléatoire
heroku create
```

**Note** : Le nom devient l'URL : `https://votre-app-clip-demo.herokuapp.com`

### ÉTAPE 11 : Déployer sur Heroku

```bash
# Ajouter remote Heroku
heroku git:remote -a votre-app-clip-demo

# Pousser sur Heroku (déclenche build automatique)
git push heroku main

# OU si votre branche s'appelle master
git push heroku master
```

**Le build va prendre 5-10 minutes** car Heroku :
1. Installe Python
2. Installe toutes les dépendances (PyTorch ~700MB)
3. Télécharge CLIP (~350MB)
4. Compile l'app

### ÉTAPE 12 : Vérifier le déploiement

```bash
# Voir les logs
heroku logs --tail

# Ouvrir l'app dans le navigateur
heroku open
```

---

## 🐛 Debugging / Problèmes courants

### Problème 1 : "Slug size too large"

**Erreur** : `Compiled slug size: 550M is too large (max is 500M)`

**Solution** :
```bash
# Vérifier taille
du -sh *

# NE PAS commiter :
# - data/Images/ (dossier images original)
# - Gros fichiers CSV
# - .venv/

# Vérifier .gitignore
cat .gitignore
```

### Problème 2 : "Memory quota exceeded (R14)"

**Erreur** : App crash avec erreur R14

**Solution** :
```bash
# Passer à Hobby dyno (1GB RAM)
heroku ps:scale web=1:hobby
```

💰 **Coût** : ~$7/mois

### Problème 3 : "Application error" au démarrage

**Solution** :
```bash
# Voir logs détaillés
heroku logs --tail

# Redémarrer l'app
heroku restart
```

**Causes fréquentes** :
- Fichiers modèles manquants dans `models/`
- Erreur dans `requirements.txt`
- Port incorrect dans Procfile

### Problème 4 : CLIP download timeout

**Solution** : C'est normal au premier lancement

CLIP télécharge automatiquement le modèle pré-entraîné (~350MB). Cela peut prendre 2-3 minutes.

```bash
# Voir la progression dans les logs
heroku logs --tail | grep -i "download"
```

### Problème 5 : ModuleNotFoundError

**Erreur** : `ModuleNotFoundError: No module named 'clip'`

**Solution** : Vérifier `requirements.txt` contient :
```
git+https://github.com/openai/CLIP.git
```

---

## 📊 Monitoring de l'app

### Voir les logs en temps réel

```bash
heroku logs --tail
```

### Voir métriques (RAM, CPU)

```bash
heroku ps
```

### Redémarrer l'app

```bash
heroku restart
```

### Arrêter l'app (économiser dynos)

```bash
heroku ps:scale web=0
```

### Relancer l'app

```bash
heroku ps:scale web=1
```

---

## 🔄 Mettre à jour l'app

Après avoir modifié le code :

```bash
# 1. Commit les changements
git add .
git commit -m "Description des changements"

# 2. Pousser sur Heroku
git push heroku main

# 3. Vérifier
heroku open
```

---

## 💰 Coûts Heroku

### Free Tier (Gratuit)
- 512 MB RAM
- Sleep après 30 min d'inactivité
- 550 heures/mois
- ⚠️ Peut être lent pour CLIP

### Hobby ($7/mois)
- 1 GB RAM
- Pas de sleep
- **Recommandé pour cette app**

### Standard ($25/mois)
- 2.5 GB RAM
- Metriques avancées
- Pour production

---

## 🎯 Checklist finale avant présentation à Karim

- [ ] App déployée et accessible en ligne
- [ ] Mode "Upload" fonctionne correctement
- [ ] Prédictions cohérentes avec métriques (94.29%)
- [ ] Interface professionnelle et claire
- [ ] Pas d'erreurs dans les logs
- [ ] URL partageable prête
- [ ] Screenshots de l'app préparés

---

## 🚀 URL à partager

Une fois déployée, votre app sera accessible à :

```
https://votre-app-clip-demo.herokuapp.com
```

Partagez cette URL avec Karim pour démonstration ! 🎉

---

## 📧 Support

**Heroku** :
- Docs : https://devcenter.heroku.com/
- Status : https://status.heroku.com/

**Streamlit** :
- Docs : https://docs.streamlit.io/
- Forum : https://discuss.streamlit.io/

**CLIP** :
- GitHub : https://github.com/openai/CLIP
- Paper : https://arxiv.org/abs/2103.00020

---

## ✅ Vous avez terminé !

Votre application CLIP est maintenant déployée sur Heroku ! 🎉

**Prochaines étapes** :
1. Tester l'app en ligne
2. Prendre screenshots pour documentation
3. Préparer démo pour Karim
4. (Optionnel) Ajouter features supplémentaires

**Bonne chance pour votre présentation chez DataSpace ! 🚀**