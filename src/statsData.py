import sqlite3
import pandas as pd
import networkx as nx

DB_PATH = "../data/PANG_Database.db"

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

def get_dataset_id(conn,dataset_name):
    cur = conn.cursor()
    cur.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
    result = cur.fetchone()
    return result[0] if result else None

def get_graphs_for_dataset(conn,dataset_id):
    query = """
    SELECT graph_index, num_nodes, num_edges, label, gspan
    FROM graphs
    WHERE dataset_id = ?
    """
    df = pd.read_sql_query(query, conn, params=(dataset_id,))
    return df

def get_filtered_patterns(conn,dataset_id, min_nodes=1, max_nodes=20, limit=100):
    """
    Récupère les motifs discriminants selon leur taille en nombre de nœuds.
    """
    query = f"""
    SELECT pattern_id, gspan, freq_total, freq_pos, freq_neg
    FROM patterns
    WHERE dataset_id = ?
    """
    df = pd.read_sql_query(query, conn, params=(dataset_id,))

    # Compter les nœuds (lignes 'v') directement dans le champ gspan
    df["num_nodes"] = df["gspan"].apply(lambda g: sum(1 for line in g.splitlines() if line.startswith("v ")))

    # Filtrage
    df = df[df["num_nodes"].between(min_nodes, max_nodes)]

    # Sélection des motifs les plus fréquents / discriminants
    df["score"] = abs(df["freq_pos"] - df["freq_neg"]) 
    df = df.sort_values("score", ascending=False).head(limit)

    return df.reset_index(drop=True)

def get_patterns_for_graph(conn,dataset_id,graph_id):
    cur = conn.cursor()
    cur.execute("""
        SELECT pattern_id FROM pattern_occurrences
        WHERE dataset_id = ? AND graph_id = ?
    """, (dataset_id, graph_id))
    present_patterns = set(row[0] for row in cur.fetchall())
    # On récupère aussi le nombre total de motifs dans le dataset
    cur.execute("SELECT COUNT(*) FROM patterns WHERE dataset_id = ?", (dataset_id,))
    total_patterns = cur.fetchone()[0]
    print(present_patterns,total_patterns)
    return present_patterns,total_patterns

def get_pattern_dict_for_dataset(conn, dataset_id):
    cur = conn.cursor()
    cur.execute("""
        SELECT pattern_id, gspan FROM patterns
        WHERE dataset_id = ?
    """, (dataset_id,))
    pattern_dict = {pattern_id: gspan for pattern_id, gspan in cur.fetchall()}
    return pattern_dict