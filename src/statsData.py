import sqlite3
import pandas as pd
import networkx as nx


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "PANG_Database.db")

def gspan_to_networkx(gspan_str):
    """
    Convertit un motif gSpan (texte) en un networkx.Graph
    """
    G = nx.Graph()
    for line in gspan_str.strip().splitlines():
        tokens = line.strip().split()
        if tokens[0] == "v":
            G.add_node(int(tokens[1]), label=tokens[2])
        elif tokens[0] == "e":
            G.add_edge(int(tokens[1]), int(tokens[2]), label=tokens[3])
    return G

def get_dataset_id(dataset_name):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
    result = cur.fetchone()
    conn.close()
    return result[0] if result else None

def get_graphs_for_dataset(dataset_id):
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT graph_index, num_nodes, num_edges, label
    FROM graphs
    WHERE dataset_id = ?
    """
    df = pd.read_sql_query(query, conn, params=(dataset_id,))
    conn.close()
    return df

def get_filtered_patterns(dataset_id, min_nodes=1, max_nodes=20, limit=100):
    """
    Récupère les motifs discriminants selon leur taille en nombre de nœuds.
    """
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT pattern_id, gspan, freq_total, freq_pos, freq_neg
    FROM patterns
    WHERE dataset_id = ?
    """
    df = pd.read_sql_query(query, conn, params=(dataset_id,))
    conn.close()

    # Compter les nœuds (lignes 'v') directement dans le champ gspan
    df["num_nodes"] = df["gspan"].apply(lambda g: sum(1 for line in g.splitlines() if line.startswith("v ")))

    # Filtrage
    df = df[df["num_nodes"].between(min_nodes, max_nodes)]

    # Sélection des motifs les plus fréquents / discriminants
    df["score"] = abs(df["freq_pos"] - df["freq_neg"]) 
    df = df.sort_values("score", ascending=False).head(limit)

    return df.reset_index(drop=True)
