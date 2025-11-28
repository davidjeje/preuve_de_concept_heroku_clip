# """
# Script pour télécharger les modèles depuis MLflow
# À exécuter AVANT de déployer sur Heroku

# Ce script récupère le modèle LightGBM et le LabelEncoder du Run MLflow spécifié.
# Il est essentiel pour alimenter le dossier 'models/' requis par 'app.py'.
# """

# import mlflow
# import os
# import shutil
# import sys # Import pour sys.exit()

# # --- CONFIGURATION (À MODIFIER PAR L'UTILISATEUR) ---
# # 🛑 L'ID du Run MLflow contenant le modèle à déployer.
# RUN_ID_TO_DEPLOY = "49582b20faa1401998ad995f3edffeff" 

# # URI du serveur MLflow. Par défaut, c'est l'instance locale.
# MLFLOW_TRACKING_URI = "http://localhost:5000"

# # Nom de l'expérience, utilisé pour lister les runs.
# MLFLOW_EXPERIMENT_NAME = "Multimodal_Classification_CLIP"
# # ----------------------------------------------------


# def download_models_from_mlflow(run_id: str, tracking_uri: str, models_dir: str = "models"):
#     """
#     Télécharge les modèles depuis MLflow et les structure dans le dossier local 'models/'.
    
#     Args:
#         run_id: ID du run MLflow contenant les modèles.
#         tracking_uri: URI du serveur MLflow.
#         models_dir: Nom du dossier de destination local.
#     """
    
#     if not run_id or run_id == "VOTRE_RUN_ID_MLFLOW_ICI":
#         print("🚨 ERREUR: Le Run ID n'est pas configuré. Veuillez le renseigner.")
#         return
        
#     print("=" * 70)
#     print("📥 TÉLÉCHARGEMENT DES MODÈLES DEPUIS MLFLOW")
#     print(f"Run ID ciblé: {run_id}")
#     print(f"Tracking URI: {tracking_uri}")
#     print("=" * 70)
    
#     # Configurer MLflow
#     mlflow.set_tracking_uri(tracking_uri)
#     client = mlflow.tracking.MlflowClient()
    
#     # --- GESTION DU DOSSIER MODELS ---
#     if os.path.exists(models_dir):
#         print(f"\n⚠️ Le dossier '{models_dir}/' existe déjà.")
#         # Utiliser input() dans un script CLI pour demander confirmation
#         response = input("Voulez-vous le remplacer (et perdre son contenu) ? (y/n): ")
#         if response.lower() == 'y':
#             shutil.rmtree(models_dir)
#             print(f"✓ Dossier supprimé.")
#         else:
#             print("❌ Opération annulée par l'utilisateur.")
#             return
    
#     os.makedirs(models_dir, exist_ok=True)
#     print(f"\n✓ Dossier '{models_dir}/' créé.")
    
#     try:
#         # 1. Lister les artifacts pour trouver les chemins
#         artifacts = client.list_artifacts(run_id)
        
#         # Le nom d'artifact par défaut pour le modèle est 'lgbm_clip_classifier'
#         LGBM_MLFLOW_PATH = "lgbm_clip_classifier" 
#         LABEL_ENCODER_MLFLOW_PATH = "label_encoder.pkl"
        
#         lgbm_artifact_found = any(a.path == LGBM_MLFLOW_PATH for a in artifacts)
#         label_encoder_artifact_found = any(a.path == LABEL_ENCODER_MLFLOW_PATH for a in artifacts)
        
        
#         # 2. Télécharger le modèle LightGBM
#         if lgbm_artifact_found:
#             print("\n📦 Téléchargement du modèle LightGBM...")
            
#             # Télécharger le dossier complet du modèle MLflow
#             temp_lgbm_path = client.download_artifacts(run_id, LGBM_MLFLOW_PATH, dst_path="temp_mlflow_downloads")
            
#             # Le fichier de poids principal se trouve dans le sous-dossier et s'appelle 'model.pkl'
#             source_pkl = os.path.join(temp_lgbm_path, "model.pkl")
#             # Destination attendue par 'app.py'
#             target_pkl = os.path.join(models_dir, "lgbm_clip_classifier.pkl")
            
#             if os.path.exists(source_pkl):
#                 shutil.copy(source_pkl, target_pkl)
#                 print(f"✓ LightGBM téléchargé et copié vers: {target_pkl}")
#             else:
#                 print("❌ Fichier 'model.pkl' non trouvé dans l'artifact MLflow. Le modèle ne sera pas utilisable.")
            
#             # Nettoyer le dossier temporaire
#             shutil.rmtree("temp_mlflow_downloads")
            
#         else:
#             print("❌ Modèle LightGBM (lgbm_clip_classifier) non trouvé dans les artifacts.")
            
#         # 3. Télécharger le label encoder
#         if label_encoder_artifact_found:
#             print("\n📦 Téléchargement du label encoder...")
            
#             # Utiliser la fonction simple de téléchargement qui place le fichier directement dans models/
#             mlflow.artifacts.download_artifacts(
#                 run_id=run_id,
#                 artifact_path=LABEL_ENCODER_MLFLOW_PATH,
#                 dst_path=models_dir
#             )
#             target_path = os.path.join(models_dir, LABEL_ENCODER_MLFLOW_PATH)
#             print(f"✓ Label encoder téléchargé: {target_path}")
#         else:
#             print("❌ Label encoder (label_encoder.pkl) non trouvé dans les artifacts.")
        
#         # 4. Vérifier les fichiers
#         print("\n" + "=" * 70)
#         print("📊 VÉRIFICATION DES FICHIERS")
#         print("=" * 70)
        
#         required_files = [
#             "lgbm_clip_classifier.pkl",
#             "label_encoder.pkl"
#         ]
        
#         all_present = True
#         for file in required_files:
#             file_path = os.path.join(models_dir, file)
#             if os.path.exists(file_path):
#                 size_mb = os.path.getsize(file_path) / (1024 * 1024)
#                 print(f"✅ {file:<30} ({size_mb:.2f} MB)")
#             else:
#                 print(f"❌ {file:<30} MANQUANT")
#                 all_present = False
        
#         if all_present:
#             print("\n" + "=" * 70)
#             print("✅ TOUS LES MODÈLES SONT PRÊTS POUR LE DÉPLOIEMENT!")
#             print("=" * 70)
#             print("\n🚀 Prochaines étapes:")
#             print("   1. Commitez les modèles: git add models/ && git commit -m 'Ajout des modèles MLflow'")
#             print("   2. Déployez votre application.")
#         else:
#             print("\n❌ Certains fichiers manquent. Vérifiez le run_id et réessayez.")
    
#     except Exception as e:
#         print(f"\n❌ ERREUR LORS DE L'OPÉRATION MLFLOW: {e}")
#         print("\n💡 Conseils:")
#         print("   - Vérifiez que MLflow est lancé (p.ex. 'poetry run mlflow ui')")
#         print("   - Vérifiez l'URI MLflow (par défaut: http://localhost:5000)")
#         print("   - Assurez-vous que le Run ID existe dans votre expérience.")


# def list_available_runs(tracking_uri: str, experiment_name: str):
#     """
#     Liste les 10 derniers runs disponibles dans MLflow.
#     """
    
#     print("=" * 70)
#     print("📋 LISTE DES RUNS MLFLOW DISPONIBLES")
#     print(f"Expérience: {experiment_name}")
#     print("=" * 70)
    
#     mlflow.set_tracking_uri(tracking_uri)
    
#     try:
#         experiment = mlflow.get_experiment_by_name(experiment_name)
        
#         if experiment is None:
#             print(f"\n❌ Expérience '{experiment_name}' non trouvée.")
#             return
        
#         runs = mlflow.search_runs(
#             experiment_ids=[experiment.experiment_id],
#             order_by=["start_time DESC"],
#             max_results=10
#         )
        
#         if len(runs) == 0:
#             print(f"\n❌ Aucun run trouvé dans l'expérience '{experiment_name}'.")
#             return
        
#         print(f"\n✓ Nombre de runs trouvés: {len(runs)}\n")
        
#         print(f"{'#':<4} {'Run ID':<35} {'Accuracy':<12} {'F1-Score':<12} {'Date':<20}")
#         print("-" * 90)
        
#         for i, run in runs.iterrows():
#             run_id = run['run_id']
#             # Utiliser .get pour gérer les métriques manquantes
#             accuracy = run.get('metrics.accuracy', 'N/A')
#             f1_score = run.get('metrics.f1_macro', 'N/A')
#             start_time = run['start_time'].strftime('%Y-%m-%d %H:%M:%S')
            
#             if isinstance(accuracy, float):
#                 accuracy = f"{accuracy:.4f}"
#             if isinstance(f1_score, float):
#                 f1_score = f"{f1_score:.4f}"
            
#             print(f"{i+1:<4} {run_id:<35} {accuracy:<12} {f1_score:<12} {start_time:<20}")
        
#         print("\n" + "=" * 70)
#         print("💡 Pour télécharger un modèle, utilisez le Run ID souhaité dans la configuration du script.")
#         print("=" * 70)
    
#     except Exception as e:
#         print(f"\n❌ ERREUR: {e}")


# if __name__ == "__main__":
    
#     print("\n🤖 SCRIPT DE TÉLÉCHARGEMENT DES MODÈLES MLFLOW\n")
    
#     # Lister les runs disponibles pour aider l'utilisateur
#     list_available_runs(MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)
    
#     print("\n" + "=" * 70)
#     print(f"⚙️ Tentative de téléchargement avec l'ID configuré: {RUN_ID_TO_DEPLOY}")
#     print("=" * 70)
    
#     # Télécharger les modèles avec l'ID configuré
#     download_models_from_mlflow(RUN_ID_TO_DEPLOY, MLFLOW_TRACKING_URI)

# 


"""
Script utilitaire pour télécharger les artefacts critiques (modèle et encodeur) 
depuis un Run MLflow spécifique.
Ceci est nécessaire pour alimenter le dossier 'models/' requis par 'app.py'.
"""
import mlflow
import os
import shutil

# --- CONFIGURATION (À MODIFIER PAR L'UTILISATEUR) ---

# 1. 🛑 IMPORTANTE MODIFICATION: REMPLACEZ CET ID par l'ID d'exécution (Run ID) du modèle que vous souhaitez déployer.
# Vous le trouvez dans l'interface MLflow.
RUN_ID_TO_DEPLOY = "49582b20faa1401998ad995f3edffeff" 

# 2. URI de suivi MLflow. 
# Si vous exécutez le serveur localement, utilisez généralement 'http://localhost:5000'.
# Si vous utilisez un serveur distant, remplacez par son adresse.
MLFLOW_TRACKING_URI = "http://localhost:5000"

# 3. Définir le dossier de destination
LOCAL_MODELS_DIR = "models"

# 4. Chemins des artefacts dans le Run MLflow
# Le modèle LightGBM est logué par mlflow.lightgbm.log_model dans un dossier
LGBM_MLFLOW_PATH = "lgbm_clip_classifier"
# Le Label Encoder est un fichier unique à la racine des artefacts
LABEL_ENCODER_MLFLOW_PATH = "label_encoder.pkl" 

# --- FONCTION DE TÉLÉCHARGEMENT ---

def download_and_prepare_artifacts():
    """Télécharge les artefacts et les place dans la structure de dossier locale."""
    if RUN_ID_TO_DEPLOY == "VOTRE_RUN_ID_MLFLOW_ICI" or RUN_ID_TO_DEPLOY == "":
        print("🚨 ERREUR: Veuillez remplacer 'VOTRE_RUN_ID_MLFLOW_ICI' par l'ID de votre Run MLflow dans le script.")
        return

    print(f"🔄 Tentative de récupération des artefacts du Run ID: {RUN_ID_TO_DEPLOY}")
    print(f"🔗 Connexion au serveur MLflow via : {MLFLOW_TRACKING_URI}")

    try:
        # Tente de définir l'URI de suivi et de vérifier l'accessibilité
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        
        # Test de base: essayer de récupérer les données du run avant le téléchargement
        run = client.get_run(RUN_ID_TO_DEPLOY)
        print(f"✓ Run ID trouvé. Nom du Run : {run.data.tags.get('mlflow.runName', 'N/A')}")

    except Exception as e:
        print(f"\n❌ ERREUR DE CONNEXION/ID MLFLOW:")
        print(f"   Vérifiez que le serveur MLflow est démarré à l'adresse {MLFLOW_TRACKING_URI}.")
        print(f"   Vérifiez que le Run ID '{RUN_ID_TO_DEPLOY}' est correct.")
        print(f"   Détails de l'erreur : {e}")
        return

    # Créer le répertoire de destination s'il n'existe pas
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
    
    # Définition des chemins cibles pour le Label Encoder et le modèle renommé
    target_pkl_model = os.path.join(LOCAL_MODELS_DIR, "lgbm_clip_classifier.pkl")
    target_pkl_le = os.path.join(LOCAL_MODELS_DIR, LABEL_ENCODER_MLFLOW_PATH)

    try:
        # --- Récupération 1: Le Label Encoder (fichier .pkl) ---
        print(f"\n-> Téléchargement du Label Encoder ({LABEL_ENCODER_MLFLOW_PATH})...")
        le_path_on_disk = mlflow.artifacts.download_artifacts(
            run_id=RUN_ID_TO_DEPLOY, 
            artifact_path=LABEL_ENCODER_MLFLOW_PATH,
            dst_path=LOCAL_MODELS_DIR
        )
        print(f"   ✓ Fichier .pkl sauvegardé localement: {target_pkl_le}")


        # --- Récupération 2: Le Modèle LightGBM (FICHIER model.pkl uniquement) ---
        # Le chemin de l'artefact est maintenant ciblé sur le fichier de poids interne
        LIGHTGBM_PKL_INTERNAL_PATH = os.path.join(LGBM_MLFLOW_PATH, "model.pkl")
        
        print(f"\n-> Téléchargement du fichier de poids LightGBM ({LIGHTGBM_PKL_INTERNAL_PATH})...")
        
        # Télécharge le fichier model.pkl dans le dossier de destination (models/)
        # Il sera initialement nommé 'model.pkl' dans le dossier 'models/'
        downloaded_pkl_path = mlflow.artifacts.download_artifacts(
            run_id=RUN_ID_TO_DEPLOY, 
            artifact_path=LIGHTGBM_PKL_INTERNAL_PATH,
            dst_path=LOCAL_MODELS_DIR
        )
        
        # L'artefact téléchargé s'appelle 'model.pkl' à la racine de LOCAL_MODELS_DIR
        temp_pkl_path = os.path.join(LOCAL_MODELS_DIR, "model.pkl")

        if os.path.exists(temp_pkl_path):
            # Renomme 'model.pkl' vers le nom attendu 'lgbm_clip_classifier.pkl'
            os.rename(temp_pkl_path, target_pkl_model)
            print(f"   ✓ Fichier de poids LightGBM téléchargé et renommé vers: {target_pkl_model}")
        else:
            print("   ⚠️ Attention: Le fichier de poids LightGBM ('model.pkl') n'a pas été trouvé après le téléchargement. L'application 'app.py' utilisera probablement le MockClassifier.")


        print("\n🎉 TERMINÉ: Les artefacts sont prêts dans le dossier 'models/' pour l'exécution de 'app.py'.")

    except Exception as e:
        # Cette erreur capture les problèmes pendant le téléchargement (ex: artifact manquant ou problème de chemin)
        print(f"\n❌ ERREUR LORS DU TÉLÉCHARGEMENT DES ARTEFACTS:")
        print(f"   Vérifiez que les chemins d'artefacts '{LIGHTGBM_PKL_INTERNAL_PATH}' et '{LABEL_ENCODER_MLFLOW_PATH}' existent dans le Run ID '{RUN_ID_TO_DEPLOY}'.")
        print(f"   Détails de l'erreur : {e}")

if __name__ == "__main__":
    download_and_prepare_artifacts()

# Note pour l'utilisateur:
# Une fois ce script exécuté, vous devriez avoir, au minimum, les fichiers suivants dans votre dossier 'models/':
# - models/label_encoder.pkl
# - models/lgbm_clip_classifier.pkl