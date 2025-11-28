votre-projet/
├── app.py                          # Application Streamlit principale
├── requirements.txt                # Dépendances Python
├── Procfile                        # Configuration Heroku
├── setup.sh                        # Configuration Streamlit
├── .gitignore                      # Fichiers à ignorer
├── models/
│   ├── lgbm_clip_classifier.pkl   # Modèle LightGBM (téléchargé depuis MLflow)
│   └── label_encoder.pkl          # Encodeur de labels
├── data/
│   └── df_clean.csv               # Dataset nettoyé (optionnel, pour mode test)
└── README.md                       # Ce fichier