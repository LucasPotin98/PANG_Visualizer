import streamlit as st
import sqlite3
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
from statsData import get_dataset_id, get_graphs_for_dataset,get_filtered_patterns,gspan_to_networkx,get_patterns_for_graph,get_pattern_dict_for_dataset
from draw import draw_graph
from predict import load_model,computePerfs,constructRepresentation,predict,get_decision_path

### Stats générales sur le dataset

# === Récupération des données
def show_dataset_page(conn,dataset_choice):
    dataset_id = get_dataset_id(conn,dataset_choice)
    df_graphs = get_graphs_for_dataset(conn,dataset_id)

    # === Statistiques générales
    st.subheader("📊 Statistiques générales")

    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de graphes", len(df_graphs))
    col2.metric("Noeuds (moy)", f"{df_graphs['num_nodes'].mean():.1f}")
    col3.metric("Arêtes (moy)", f"{df_graphs['num_edges'].mean():.1f}")

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("#### Distribution du nombre de nœuds")
        fig, ax = plt.subplots()
        ax.hist(df_graphs['num_nodes'], bins=20, color="skyblue", edgecolor="black")
        ax.set_xlabel("Nombre de nœuds")
        ax.set_ylabel("Nombre de graphes")
        st.pyplot(fig)

    with col5:
        st.markdown("#### Répartition des classes")
        st.bar_chart(df_graphs['label'].value_counts().sort_index())


        # Step 2: Pattern selection and configuration
    st.subheader("🧩 Extraction de motifs discriminants")
    col1, col2, col3 = st.columns(3)
    with col1:
        s = st.slider("Nombre de motifs discriminants (s)", 10, 500, 100)
    with col2:
        min_nodes = st.slider("Nœuds minimum par motif", 1, 10, 3)
    with col3:
        max_nodes = st.slider("Nœuds maximum par motif", 5, 20, 10)


    df_patterns = get_filtered_patterns(conn,
        dataset_id=dataset_id,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        limit=s
        )

        # Affichage du tableau
    st.markdown("Exemples de motifs discriminants :")
    st.dataframe(df_patterns[["pattern_id", "freq_total", "freq_pos", "freq_neg", "num_nodes"]].head(5))

    # Sélection du motif à afficher
    pattern_options = df_patterns["pattern_id"].tolist()
    if pattern_options:
        selected_index = st.selectbox("🎯 Motif à visualiser :", pattern_options)
        selected_gspan = df_patterns[df_patterns["pattern_id"] == selected_index]["gspan"].values[0]

        # Conversion gSpan → networkx
        G = gspan_to_networkx(selected_gspan)

        fig = draw_graph(G, dataset_choice,figsize=(5, 5))
        st.pyplot(fig)
    else:
        st.warning("Aucun motif à afficher.")

def show_graph_analysis_page(conn,dataset_choice):
    dataset_id = get_dataset_id(conn,dataset_choice)
    df_graphs = get_graphs_for_dataset(conn,dataset_id)
    model = load_model(dataset_choice)
    patternsDict = get_pattern_dict_for_dataset(conn,dataset_id)

     # === 2. Afficher les performances (codées pour l’instant)
    st.markdown("### 📈 Performances du modèle")
    mean, std = computePerfs(model, dataset_choice)
    st.write(f"F1-score moyen (CV) : **{mean:.2f}** (± {std:.2f})")

    # === 3. Sélection du graphe
    st.markdown("### 🧪 Sélectionnez un graphe à analyser")
    graph_options = df_graphs['graph_index'].tolist()
    selected_graph = st.selectbox("Choisissez un identifiant de graphe :", graph_options)
    
    selected_gspan = df_graphs[df_graphs["graph_index"] == selected_graph]["gspan"].values[0]
    G = gspan_to_networkx(selected_gspan)
    fig = draw_graph(G, dataset_choice, figsize=(5, 5))
    st.pyplot(fig)

    # === 4. Prédiction
    st.markdown("### 🎯 Prédiction")
    selected_patterns,total = get_patterns_for_graph(conn, dataset_id, selected_graph)
    representation = constructRepresentation(selected_patterns, total)
    
    prediction = predict(model, representation)
    st.write(f"Le modèle prédit la classe suivante : **{prediction[0]}**") 

    # === 5. Interprétation des motifs
    st.markdown("### 💡 Sous-graphes discriminants")
    patterns = get_decision_path(model, representation)
    if patterns:
        #separer en 5 colonnes
        col1, col2, col3, col4 = st.columns(4)
        for i, (feature, value) in enumerate(patterns):
            with locals()[f"col{i % 4 + 1}"]:
                st.markdown(f"**Motif {i+1}**")
                G = gspan_to_networkx(patternsDict[feature])
                fig = draw_graph(G, dataset_choice, figsize=(3, 3))
                st.pyplot(fig)
                st.markdown(f"<div style='text-align: center;'>{'Motif Présent' if value == 1 else 'Motif Absent'}</div>", unsafe_allow_html=True)
    else:
        st.warning("Aucun motif discriminant trouvé.")

    

# LOAD LA DATABASE
conn = sqlite3.connect("../data/PANG_Database.db")
# Sidebar: Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Selection de Page", ["Exploration du dataset", "Analyse par graphe"])
st.sidebar.markdown("---")

# Sidebar: Dataset selection
dataset_choice = st.sidebar.selectbox("Choisissez un dataset", ["MUTAG", "PTC", "AIDS"])
st.sidebar.markdown("---")

# Dataset description
dataset_descriptions = {
    "MUTAG": "MUTAG est un dataset de graphes représentant des composés chimiques. Chaque graphe est étiqueté selon la mutagénicité du composé.",
    "PTC": "PTC est un dataset de graphes représentant des composés chimiques. Chaque graphe est étiqueté selon la toxicité du composé.",   
    "AIDS": "AIDS est un dataset de graphes représentant des composés chimiques. Chaque graphe est étiqueté selon la présence de virus du SIDA"
}

st.markdown(dataset_descriptions[dataset_choice])


# Page content
if page == "Exploration du dataset":
    show_dataset_page(conn,dataset_choice)
elif page == "Analyse par graphe":
    show_graph_analysis_page(conn,dataset_choice)
