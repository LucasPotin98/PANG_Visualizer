import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

# Title
st.title("🔍 PANG: Pattern-based Anomaly Detection in Graphs")
st.markdown("""
Cette application illustre le fonctionnement de **PANG** sur des jeux de données de graphes classiques.

Choisissez un dataset, explorez les motifs discriminants extraits, vectorisez les graphes et entraînez un classifieur !
""")

# Sidebar: Dataset selection
dataset_choice = st.sidebar.selectbox("Choisissez un dataset", ["MUTAG", "PTC", "NCI1", "D&D"])
st.sidebar.markdown("---")

# Dataset description
dataset_descriptions = {
    "MUTAG": "MUTAG est un dataset de graphes représentant des composés chimiques. Chaque graphe est étiqueté selon la mutagénicité du composé.",
    "PTC": "PTC est un dataset de graphes représentant des composés chimiques. Chaque graphe est étiqueté selon la toxicité du composé.",
    "NCI1": "NCI1 est un dataset de graphes représentant des composés chimiques. Chaque graphe est étiqueté selon l'activité biologique du composé.",
    "D&D": "D&D est un dataset de graphes représentant des structures de protéines. Chaque graphe est étiqueté selon la fonction biologique de la protéine."
}

st.markdown(dataset_descriptions[dataset_choice])

# Placeholder for dataset loading (to be replaced with actual loaders)
st.subheader(f"🗂 Aperçu du dataset {dataset_choice}")
st.markdown("*(Chargement simulé — à remplacer par des loaders TUDataset ou autres)*")

# Simulated graph stats
data_stats = {
    "MUTAG": {"graphs": 188, "nodes": 18, "edges": 20, "classes": 2},
    "PTC": {"graphs": 350, "nodes": 26, "edges": 26, "classes": 2},
    "NCI1": {"graphs": 4110, "nodes": 30, "edges": 32, "classes": 2},
    "D&D": {"graphs": 1178, "nodes": 285, "edges": 716, "classes": 2},
}
st.write(pd.DataFrame([data_stats[dataset_choice]]).T.rename(columns={0: "Valeur"}))

# Quelques stats generales

# Step 2: Pattern selection and configuration
st.subheader("🧩 Extraction de motifs discriminants")
pattern_type = st.selectbox("Type de motifs", ["Généraux", "Induits", "Fermés"], index=1)
s = st.slider("Nombre de motifs discriminants (s)", 10, 500, 100)

st.markdown("Exemples simulés de motifs discriminants :")
col1, col2 = st.columns(2)
with col1:
    G1 = nx.cycle_graph(4)
    nx.draw(G1, with_labels=True)
    st.pyplot(plt.gcf())
    plt.clf()
with col2:
    G2 = nx.path_graph(5)
    nx.draw(G2, with_labels=True)
    st.pyplot(plt.gcf())
    plt.clf()

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
