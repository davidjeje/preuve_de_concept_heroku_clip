
"""
Application Streamlit pour tester le modèle CLIP + LightGBM
Classification multimodale de produits e-commerce
OPTIMISÉ POUR HEROKU : Charge les images depuis S3 à la demande
"""

import streamlit as st
import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import os
import boto3
from io import BytesIO

# =============================
# 1. CONFIGURATION DE LA PAGE
# =============================
st.set_page_config(
    page_title="Classification Produits CLIP",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .prediction-box { background-color: #e8f4f8; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; margin: 1rem 0; }
    </style>
""", unsafe_allow_html=True)

# =============================
# 2. CONFIGURATION S3
# =============================
BUCKET_NAME = "preuve-de-concept"

@st.cache_resource
def init_s3_client():
    """Initialise le client S3 une seule fois"""
    # Priorité aux variables d'environnement
    aws_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_region = os.environ.get('AWS_DEFAULT_REGION', 'eu-west-3')
    
    if aws_key and aws_secret:
        # Heroku : utiliser les variables d'environnement
        return boto3.client(
            's3',
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=aws_region
        )
    else:
        # Local : utiliser st.secrets (fallback)
        try:
            return boto3.client(
                's3',
                aws_access_key_id=st.secrets["default"]["aws_access_key_id"],
                aws_secret_access_key=st.secrets["default"]["aws_secret_access_key"],
                region_name=st.secrets["default"]["region"]
            )
        except Exception as e:
            st.error("❌ Erreur : Credentials AWS non configurés")
            st.info("""
            **Configuration requise :**
            - **Heroku** : `heroku config:set AWS_ACCESS_KEY_ID=XXX`
            - **Local** : Fichier `.streamlit/secrets.toml`
            """)
            st.stop()

def download_s3_folder(s3_client, bucket, prefix, local_dir):
    """Télécharge un dossier S3 vers le disque local"""
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('/'):
                continue
            
            local_path = os.path.join(local_dir, os.path.relpath(key, prefix))
            local_folder = os.path.dirname(local_path)
            os.makedirs(local_folder, exist_ok=True)
            
            s3_client.download_file(bucket, key, local_path)
            print(f"✓ Téléchargé: {key}")

@st.cache_resource
def download_models_and_csv():
    """Télécharge UNIQUEMENT les modèles et le CSV (pas les images)"""
    s3 = init_s3_client()
    
    # Télécharger modèles (petit, ~quelques MB)
    if not os.path.exists("models/lgbm_clip_classifier.pkl"):
        with st.spinner("📥 Téléchargement des modèles..."):
            download_s3_folder(s3, BUCKET_NAME, "models", "models")
    
    # Télécharger uniquement le CSV (petit)
    if not os.path.exists("data/df_clean.csv"):
        with st.spinner("📥 Téléchargement du dataset CSV..."):
            os.makedirs("data", exist_ok=True)
            s3.download_file(BUCKET_NAME, "data/df_clean.csv", "data/df_clean.csv")
    
    return True

@st.cache_data(ttl=3600)  # Cache pendant 1 heure
def load_image_from_s3(image_path):
    """
    Charge une image depuis S3 à la demande
    Cache le résultat pour éviter de re-télécharger la même image
    """
    s3 = init_s3_client()
    
    try:
        # Normaliser le chemin
        s3_key = image_path.replace("\\", "/")
        
        # Si le chemin commence par "data/Images/", le garder tel quel
        # Sinon, ajouter le préfixe
        if not s3_key.startswith("data/Images/"):
            s3_key = f"data/Images/{s3_key}"
        
        # Télécharger l'image en mémoire (pas sur disque)
        buffer = BytesIO()
        s3.download_fileobj(BUCKET_NAME, s3_key, buffer)
        buffer.seek(0)
        
        # Ouvrir l'image
        image = Image.open(buffer).convert("RGB")
        return image
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de l'image depuis S3: {e}")
        st.info(f"Chemin S3 tenté: {s3_key}")
        return None

# =============================
# 3. CHARGEMENT DES MODÈLES
# =============================
@st.cache_resource
def load_models():
    """Charge CLIP et LightGBM"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        
        lgbm_model = joblib.load("models/lgbm_clip_classifier.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        
        return clip_model, clip_preprocess, lgbm_model, label_encoder, device
    
    except FileNotFoundError:
        st.error("❌ Fichiers modèles introuvables")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des modèles: {e}")
        st.stop()

# =============================
# 4. FONCTIONS PRÉDICTION
# =============================
def extract_clip_embedding(image, text, clip_model, clip_preprocess, device):
    try:
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        text_input = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            embedding = torch.cat((image_features, text_features), dim=1)
        return embedding.cpu().numpy()
    except Exception as e:
        st.error(f"❌ Erreur lors de l'extraction des embeddings: {e}")
        return None

def predict_category(embedding, lgbm_model, label_encoder):
    try:
        prediction = lgbm_model.predict(embedding)[0]
        probabilities = lgbm_model.predict_proba(embedding)[0]
        category = label_encoder.inverse_transform([prediction])[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_categories = label_encoder.inverse_transform(top_3_indices)
        top_3_probs = probabilities[top_3_indices]
        return category, probabilities[prediction], top_3_categories, top_3_probs
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction: {e}")
        return None, None, None, None

def test_product(sample, clip_model, clip_preprocess, lgbm_model, label_encoder, device):
    st.markdown("---")
    st.header("📦 Produit testé")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Image")
        with st.spinner("📥 Chargement de l'image depuis S3..."):
            # Charger l'image depuis S3 à la demande
            image = load_image_from_s3(sample['image_path'])
            
            if image is None:
                st.error("❌ Impossible de charger l'image")
                return
            
            st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("📝 Informations")
        st.markdown(f"**Texte**: {sample['text_clip'][:200]}...\n\n**Vraie catégorie**: `{sample['category']}`")
    
    with st.spinner("🔄 Prédiction en cours..."):
        embedding = extract_clip_embedding(image, sample['text_clip'], clip_model, clip_preprocess, device)
        if embedding is not None:
            category, confidence, top_3_cats, top_3_probs = predict_category(embedding, lgbm_model, label_encoder)
            if category is not None:
                st.markdown("---")
                is_correct = category == sample['category']
                if is_correct:
                    st.success(f"✅ **CORRECT** - Prédiction: `{category}` (Confiance: {confidence:.2%})")
                else:
                    st.error(f"❌ **ERREUR** - Prédiction: `{category}` | Vrai: `{sample['category']}` (Confiance: {confidence:.2%})")
                st.subheader("📈 Top 3 des prédictions")
                for i, (cat, prob) in enumerate(zip(top_3_cats, top_3_probs), 1):
                    emoji = "✅" if cat == sample['category'] else ""
                    st.write(f"{i}. **{cat}** {emoji}: {prob:.2%}")

# =============================
# 5. APPLICATION PRINCIPALE
# =============================
def main():
    # Télécharger uniquement modèles + CSV (pas les images)
    download_models_and_csv()
    
    st.markdown('<h1 class="main-header">🛍️ Classification de Produits avec CLIP</h1>', unsafe_allow_html=True)
    st.markdown("""
    Cette application utilise **CLIP (Vision Transformer)** + **LightGBM** 
    pour classifier des produits e-commerce.
    
    **Performance** : Accuracy 94.29% | F1-Score 94.26%
    
    💡 *Les images sont chargées depuis AWS S3 à la demande pour optimiser la mémoire.*
    """)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        st.subheader("📊 Modèle")
        st.info("""
        **Architecture**: CLIP ViT-B/32  
        **Classifieur**: LightGBM  
        **Embeddings**: 1024-d (512 image + 512 texte)
        """)
        st.markdown("---")
        test_mode = st.radio("Mode de test", ["🖼️ Upload Image + Texte", "📊 Tester sur Dataset"])
        st.markdown("---")
        device_used = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.metric("Device", device_used)
    
    with st.spinner("🔄 Chargement des modèles..."):
        clip_model, clip_preprocess, lgbm_model, label_encoder, device = load_models()
    st.success("✅ Modèles chargés avec succès!")
    
    with st.expander("📋 Catégories disponibles"):
        categories = label_encoder.classes_
        st.write(f"**Nombre de catégories**: {len(categories)}")
        cols = st.columns(3)
        for i, cat in enumerate(categories):
            cols[i % 3].write(f"• {cat}")
    
    st.markdown("---")
    
    if test_mode == "🖼️ Upload Image + Texte":
        st.header("🖼️ Test avec vos propres données")
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Image uploadée", use_column_width=True)
        
        with col2:
            product_name = st.text_input("Nom du produit", placeholder="Ex: Nike Air Max Sneakers")
            product_desc = st.text_area("Description détaillée", placeholder="Ex: Chaussures de sport...")
            full_text = f"{product_name} {product_desc}".strip()
            if full_text:
                st.info(f"**Texte combiné** ({len(full_text)} caractères): {full_text[:100]}...")
        
        st.markdown("---")
        if st.button("🚀 Prédire la catégorie"):
            if uploaded_file is None:
                st.warning("⚠️ Veuillez uploader une image")
            elif not full_text:
                st.warning("⚠️ Veuillez entrer au moins le nom du produit")
            else:
                with st.spinner("🔄 Analyse en cours..."):
                    embedding = extract_clip_embedding(image, full_text, clip_model, clip_preprocess, device)
                    if embedding is not None:
                        category, confidence, top_3_cats, top_3_probs = predict_category(embedding, lgbm_model, label_encoder)
                        if category is not None:
                            st.markdown("---")
                            st.header("📊 Résultats de la prédiction")
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h2 style="color: #1f77b4; margin-top: 0;">🎯 Catégorie prédite</h2>
                                <h1 style="margin: 0.5rem 0;">{category}</h1>
                                <p style="font-size: 1.2rem; margin: 0;">Confiance: <strong>{confidence:.2%}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.subheader("📈 Top 3 des prédictions")
                            for i, (cat, prob) in enumerate(zip(top_3_cats, top_3_probs), 1):
                                col1, col2, col3 = st.columns([1, 4, 2])
                                with col1: st.markdown("🥇" if i==1 else "🥈" if i==2 else "🥉")
                                with col2: st.markdown(f"**{cat}**")
                                with col3: st.progress(float(prob)); st.caption(f"{prob:.2%}")
    
    else:
        st.header("📊 Test sur le dataset")
        try:
            df_clean = pd.read_csv("data/df_clean.csv")
            st.success(f"✅ Dataset chargé: {len(df_clean)} produits")
            selection_mode = st.radio("Mode de sélection", ["🎲 Aléatoire", "🔍 Sélection manuelle"])
            
            if selection_mode == "🎲 Aléatoire":
                if st.button("🎲 Tirer un produit aléatoire"):
                    sample = df_clean.sample(1).iloc[0]
                    test_product(sample, clip_model, clip_preprocess, lgbm_model, label_encoder, device)
            else:
                index = st.number_input("Index du produit", min_value=0, max_value=len(df_clean)-1, value=0)
                if st.button("🔍 Tester ce produit"):
                    sample = df_clean.iloc[index]
                    test_product(sample, clip_model, clip_preprocess, lgbm_model, label_encoder, device)
        
        except FileNotFoundError:
            st.error("❌ Fichier 'data/df_clean.csv' introuvable")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>🤖 Développé avec CLIP (OpenAI) + LightGBM | 📊 94.29% Accuracy | ☁️ Images hébergées sur AWS S3</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()