import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
from statsData import get_dataset_id, get_graphs_for_dataset,get_filtered_patterns,gspan_to_networkx
from draw import draw_graph

# Title
st.title("🕸️  PANG Visualizer")
st.markdown("""
Cette application illustre le fonctionnement de **PANG** sur des jeux de données de graphes classiques.

Choisissez un dataset, explorez les motifs discriminants extraits, vectorisez les graphes et entraînez un classifieur !
""")

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


### Stats générales sur le dataset

# === Récupération des données
dataset_id = get_dataset_id(dataset_choice)
df_graphs = get_graphs_for_dataset(dataset_id)

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


df_patterns = get_filtered_patterns(
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

    fig = draw_graph(G, dataset_choice)
    st.pyplot(fig)
else:
    st.warning("Aucun motif à afficher.")

# Step 3: Vectorization (simulated)
st.subheader("🔢 Vectorisation des graphes")
st.markdown("Vectorisation simulée. Chaque graphe est représenté par un vecteur de dimension `s`.")

X = np.random.randint(0, 3, size=(100, s))
y = np.random.randint(0, 2, size=100)

st.dataframe(pd.DataFrame(X[:10], columns=[f"P{i}" for i in range(1, s+1)]).assign(Label=y[:10]))

# Step 4: Classification
st.subheader("🧠 Entraînement du classifieur")
classifier_name = st.selectbox("Choisir un classifieur", ["SVM", "Random Forest"])

if classifier_name == "SVM":
    clf = SVC(kernel='linear')
else:
    clf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
st.success(f"F-score moyen (5-fold CV) : {scores.mean():.2f} ± {scores.std():.2f}")

# Footer
st.markdown("---")
st.markdown("Application pédagogique basée sur le framework PANG \[Potin et al., ECML PKDD 2023\]")
