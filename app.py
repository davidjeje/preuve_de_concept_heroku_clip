"""
Application Streamlit pour tester le modèle CLIP + LightGBM
Classification multimodale de produits e-commerce
"""

import streamlit as st
import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import os
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="Classification Produits CLIP",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Cache pour charger les modèles (éviter rechargement à chaque interaction)
@st.cache_resource
def load_models():
    """Charge CLIP et le classifieur LightGBM"""
    try:
        # Charger CLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        
        # Charger LightGBM et label encoder
        lgbm_model = joblib.load("models/lgbm_clip_classifier.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        
        return clip_model, clip_preprocess, lgbm_model, label_encoder, device
    
    except FileNotFoundError as e:
        st.error(f"❌ Erreur: Fichiers modèles introuvables. Assurez-vous que 'models/' contient les fichiers .pkl")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des modèles: {e}")
        st.stop()


def extract_clip_embedding(image, text, clip_model, clip_preprocess, device):
    """Extrait l'embedding CLIP d'une image et d'un texte"""
    try:
        # Prétraiter l'image
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        
        # Tokeniser le texte (avec troncation)
        text_input = clip.tokenize([text], truncate=True).to(device)
        
        # Extraire les embeddings
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            
            # Normaliser
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Concatener
            embedding = torch.cat((image_features, text_features), dim=1)
        
        return embedding.cpu().numpy()
    
    except Exception as e:
        st.error(f"❌ Erreur lors de l'extraction des embeddings: {e}")
        return None


def predict_category(embedding, lgbm_model, label_encoder):
    """Prédit la catégorie avec probabilités"""
    try:
        # Prédiction
        prediction = lgbm_model.predict(embedding)[0]
        probabilities = lgbm_model.predict_proba(embedding)[0]
        
        # Décoder la catégorie
        category = label_encoder.inverse_transform([prediction])[0]
        
        # Top 3 catégories avec probabilités
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_categories = label_encoder.inverse_transform(top_3_indices)
        top_3_probs = probabilities[top_3_indices]
        
        return category, probabilities[prediction], top_3_categories, top_3_probs
    
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction: {e}")
        return None, None, None, None


def main():
    """Application principale"""
    
    # En-tête
    st.markdown('<h1 class="main-header">🛍️ Classification de Produits avec CLIP</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Cette application utilise le modèle **CLIP (Vision Transformer)** combiné à **LightGBM** 
    pour classifier automatiquement des produits e-commerce à partir d'une image et d'une description.
    
    **Performance du modèle** : Accuracy 94.29% | F1-Score 94.26%
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        
        # Informations modèle
        st.subheader("📊 Modèle")
        st.info("""
        **Architecture**: CLIP ViT-B/32  
        **Classifieur**: LightGBM  
        **Embeddings**: 1024-d (512 image + 512 texte)  
        **Dataset**: Flipkart E-commerce
        """)
        
        st.markdown("---")
        
        # Mode de test
        test_mode = st.radio(
            "Mode de test",
            ["🖼️ Upload Image + Texte", "📊 Tester sur Dataset"],
            help="Choisissez comment tester le modèle"
        )
        
        st.markdown("---")
        
        # Infos techniques
        device_used = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.metric("Device", device_used)
    
    # Charger les modèles
    with st.spinner("🔄 Chargement des modèles..."):
        clip_model, clip_preprocess, lgbm_model, label_encoder, device = load_models()
    
    st.success("✅ Modèles chargés avec succès!")
    
    # Afficher les catégories disponibles
    with st.expander("📋 Catégories disponibles"):
        categories = label_encoder.classes_
        st.write(f"**Nombre de catégories**: {len(categories)}")
        
        # Afficher en colonnes
        cols = st.columns(3)
        for i, cat in enumerate(categories):
            cols[i % 3].write(f"• {cat}")
    
    st.markdown("---")
    
    # Mode 1: Upload manuel
    if test_mode == "🖼️ Upload Image + Texte":
        st.header("🖼️ Test avec vos propres données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Image du produit")
            uploaded_file = st.file_uploader(
                "Choisir une image...", 
                type=["jpg", "jpeg", "png"],
                help="Format: JPG, JPEG, PNG"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Image uploadée", use_column_width=True)
        
        with col2:
            st.subheader("📝 Description du produit")
            
            product_name = st.text_input(
                "Nom du produit",
                placeholder="Ex: Nike Air Max Sneakers",
                help="Nom court du produit"
            )
            
            product_desc = st.text_area(
                "Description détaillée",
                placeholder="Ex: Chaussures de sport confortables avec semelle air...",
                height=150,
                help="Description complète du produit"
            )
            
            # Combiner nom + description
            full_text = f"{product_name} {product_desc}".strip()
            
            if full_text:
                st.info(f"**Texte combiné** ({len(full_text)} caractères): {full_text[:100]}...")
        
        # Bouton de prédiction
        st.markdown("---")
        
        if st.button("🚀 Prédire la catégorie", type="primary", use_container_width=True):
            if uploaded_file is None:
                st.warning("⚠️ Veuillez uploader une image")
            elif not full_text:
                st.warning("⚠️ Veuillez entrer au moins le nom du produit")
            else:
                with st.spinner("🔄 Analyse en cours..."):
                    # Extraire embeddings
                    embedding = extract_clip_embedding(
                        image, full_text, clip_model, clip_preprocess, device
                    )
                    
                    if embedding is not None:
                        # Prédire
                        category, confidence, top_3_cats, top_3_probs = predict_category(
                            embedding, lgbm_model, label_encoder
                        )
                        
                        if category is not None:
                            # Afficher résultats
                            st.markdown("---")
                            st.header("📊 Résultats de la prédiction")
                            
                            # Prédiction principale
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h2 style="color: #1f77b4; margin-top: 0;">🎯 Catégorie prédite</h2>
                                <h1 style="margin: 0.5rem 0;">{category}</h1>
                                <p style="font-size: 1.2rem; margin: 0;">
                                    Confiance: <strong>{confidence:.2%}</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Top 3 prédictions
                            st.subheader("📈 Top 3 des prédictions")
                            
                            for i, (cat, prob) in enumerate(zip(top_3_cats, top_3_probs), 1):
                                col1, col2, col3 = st.columns([1, 4, 2])
                                
                                with col1:
                                    if i == 1:
                                        st.markdown("🥇")
                                    elif i == 2:
                                        st.markdown("🥈")
                                    else:
                                        st.markdown("🥉")
                                
                                with col2:
                                    st.markdown(f"**{cat}**")
                                
                                with col3:
                                    st.progress(float(prob))
                                    st.caption(f"{prob:.2%}")
    
    # Mode 2: Tester sur dataset
    else:
        st.header("📊 Test sur le dataset")
        
        # Charger le dataset
        try:
            df_clean = pd.read_csv("data/df_clean.csv")
            st.success(f"✅ Dataset chargé: {len(df_clean)} produits")
            
            # Sélection aléatoire ou manuelle
            selection_mode = st.radio(
                "Mode de sélection",
                ["🎲 Aléatoire", "🔍 Sélection manuelle"]
            )
            
            if selection_mode == "🎲 Aléatoire":
                if st.button("🎲 Tirer un produit aléatoire", type="primary"):
                    sample = df_clean.sample(1).iloc[0]
                    test_product(sample, clip_model, clip_preprocess, lgbm_model, label_encoder, device)
            
            else:
                # Sélection par index
                index = st.number_input(
                    "Index du produit",
                    min_value=0,
                    max_value=len(df_clean)-1,
                    value=0
                )
                
                if st.button("🔍 Tester ce produit", type="primary"):
                    sample = df_clean.iloc[index]
                    test_product(sample, clip_model, clip_preprocess, lgbm_model, label_encoder, device)
        
        except FileNotFoundError:
            st.error("❌ Fichier 'data/df_clean.csv' introuvable")
            st.info("💡 Créez ce fichier en exportant votre dataset nettoyé avec: `df_clean.to_csv('data/df_clean.csv', index=False)`")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>🤖 Développé avec CLIP (OpenAI) + LightGBM | 
        📊 Performance: 94.29% Accuracy | 
        🚀 Déployé sur Heroku</p>
    </div>
    """, unsafe_allow_html=True)


def test_product(sample, clip_model, clip_preprocess, lgbm_model, label_encoder, device):
    """Teste un produit du dataset"""
    
    st.markdown("---")
    st.header("📦 Produit testé")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Image")
        try:
            image = Image.open(sample['image_path']).convert("RGB")
            st.image(image, use_column_width=True)
        except Exception as e:
            st.error(f"❌ Impossible de charger l'image: {e}")
            return
    
    with col2:
        st.subheader("📝 Informations")
        st.markdown(f"""
        **Texte**: {sample['text_clip'][:200]}...
        
        **Vraie catégorie**: `{sample['category']}`
        """)
    
    # Prédiction
    with st.spinner("🔄 Prédiction en cours..."):
        embedding = extract_clip_embedding(
            image, sample['text_clip'], clip_model, clip_preprocess, device
        )
        
        if embedding is not None:
            category, confidence, top_3_cats, top_3_probs = predict_category(
                embedding, lgbm_model, label_encoder
            )
            
            if category is not None:
                st.markdown("---")
                
                # Vérifier si prédiction correcte
                is_correct = category == sample['category']
                
                if is_correct:
                    st.success(f"✅ **CORRECT** - Prédiction: `{category}` (Confiance: {confidence:.2%})")
                else:
                    st.error(f"❌ **ERREUR** - Prédiction: `{category}` | Vrai: `{sample['category']}` (Confiance: {confidence:.2%})")
                
                # Top 3
                st.subheader("📈 Top 3 des prédictions")
                for i, (cat, prob) in enumerate(zip(top_3_cats, top_3_probs), 1):
                    emoji = "✅" if cat == sample['category'] else ""
                    st.write(f"{i}. **{cat}** {emoji}: {prob:.2%}")


if __name__ == "__main__":
    main()