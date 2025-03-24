import sqlite3
import networkx as nx
import io
import re

conn = sqlite3.connect("../data/PANG_Database.db")
cur = conn.cursor()

# Suppression des tables si elles existent
cur.execute("DROP TABLE IF EXISTS graphs")
cur.execute("DROP TABLE IF EXISTS datasets")
cur.execute("DROP TABLE IF EXISTS pattern_occurrences")
cur.execute("DROP TABLE IF EXISTS patterns")

# Création de la table datasets
cur.execute("""
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT
)
""")

# Création de la table graphs
cur.execute("""
CREATE TABLE IF NOT EXISTS graphs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL ,
    graph_index INTEGER NOT NULL, 
    num_nodes INTEGER NOT NULL,
    num_edges INTEGER NOT NULL,
    label INTEGER,                         
    gspan TEXT NOT NULL                 
)
""")

cur.execute("""
CREATE TABLE patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    pattern_id INTEGER NOT NULL,
    gspan TEXT NOT NULL,
    ranking_patterns TEXT,
    freq_total INTEGER NOT NULL,
    freq_pos INTEGER NOT NULL,
    freq_neg INTEGER NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
)
""")

cur.execute("""
CREATE TABLE pattern_occurrences (
    dataset_id INTEGER NOT NULL,
    pattern_id INTEGER NOT NULL,
    graph_id INTEGER NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (dataset_id,pattern_id, graph_id)
)
""")

# Insertion des datasets benchmark
datasets = [
    ("MUTAG", "Graphes moléculaires avec label de mutagénicité"),
    ("PTC", "Graphes moléculaires avec label de toxicité"),
    ('AIDS', 'Graphes moléculaires avec label de présence de virus du SIDA')]

cur.executemany("INSERT OR IGNORE INTO datasets (name, description) VALUES (?, ?)", datasets)

def parse_gspan_string(gspan_str):
    lines = gspan_str.strip().splitlines()
    G = nx.Graph()
    for line in lines:
        tokens = line.split()
        if tokens[0] == 'v':
            G.add_node(int(tokens[1]), label=tokens[2])
        elif tokens[0] == 'e':
            G.add_edge(int(tokens[1]), int(tokens[2]), label=tokens[3])
    return G

def read_gspan_graphs(file_path,labels,dataset_id):
    graphs = []
    with open(file_path, 'r') as f:
        content = f.read()

    # Split sur les lignes t # id
    raw_graphs = re.split(r'^t\s+#\s+\d+', content, flags=re.MULTILINE)[1:]
    graph_ids = [int(m.group(1)) for m in re.finditer(r'^t\s+#\s+(\d+)', content, flags=re.MULTILINE)]

    for gid, raw in zip(graph_ids, raw_graphs):
        lines = raw.strip().splitlines()
        num_nodes = sum(1 for l in lines if l.startswith("v "))
        num_edges = sum(1 for l in lines if l.startswith("e "))
        graphs.append((dataset_id,gid, num_nodes, num_edges, labels[gid], raw.strip()))

    return graphs

def read_labels(file_path):
    """
    Lit un fichier de labels ligne par ligne et les convertit en deux classes (0/1).
    - Les labels <= 0 deviennent 0
    - Les labels >= 1 deviennent 1
    """
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            label = int(line.strip().split("\t")[0])
            labels.append(0 if label <= 0 else 1)
    return labels

def read_patterns(file_path, labels, dataset_id):
    """
    Lit un fichier de motifs gSpan enrichi et renvoie deux éléments :
    1. Liste des motifs : (dataset_id, pattern_index, freq_total, freq_pos, freq_neg, gspan_str)
    2. Liste des occurrences : (dataset_id, pattern_index, graph_index)
    """
    patterns = []
    occurrences = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    current_id = None
    buffer = []
    graph_ids = []

    for line in lines:
        line = line.strip()
        if line.startswith("t #"):
            if current_id is not None and buffer:
                gspan_str = "\n".join(buffer)
                freq_total = len(graph_ids)
                freq_pos = sum(1 for gid in graph_ids if labels[gid] == 1)
                freq_neg = sum(1 for gid in graph_ids if labels[gid] == 0)
                patterns.append((dataset_id, current_id, freq_total, freq_pos, freq_neg, gspan_str))
                for gid in graph_ids:
                    occurrences.append((dataset_id, current_id, gid))
            current_id = int(line.split("#")[1].split("*")[0].strip())
            buffer = []
            graph_ids = []
        elif line.startswith("x "):
            graph_ids = list(map(int, line[2:].strip().split()))
        else:
            buffer.append(line)

    # Dernier motif
    if current_id is not None and buffer:
        gspan_str = "\n".join(buffer)
        freq_total = len(graph_ids)
        freq_pos = sum(1 for gid in graph_ids if labels[gid] == 1)
        freq_neg = sum(1 for gid in graph_ids if labels[gid] == 0)
        patterns.append((dataset_id, current_id, freq_total, freq_pos, freq_neg, gspan_str))
        for gid in graph_ids:
            occurrences.append((dataset_id, current_id, gid))

    return patterns, occurrences

c=1
for ds in ["MUTAG", "PTC", "AIDS"]:
    file = "../data/"+ds+"_graph.txt"
    fileLabels = "../data/"+ds+"_label.txt"
    filePattern = "../data/"+ds+"_pattern.txt"
    labels = read_labels(fileLabels)
    graph_data = read_gspan_graphs(file,labels,c)
    patterns, occurrences = read_patterns(filePattern, labels,c)
    print(graph_data)

    # Insertion
    cur.executemany("""
    INSERT OR REPLACE INTO graphs (dataset_id,graph_index, num_nodes, num_edges, label, gspan)
    VALUES (? , ?, ?, ?, ?, ?)
    """, graph_data)

    cur.executemany("""
    INSERT INTO patterns (dataset_id, pattern_id, freq_total, freq_pos, freq_neg, gspan)
    VALUES (?, ?, ?, ?, ?, ?)
    """, patterns)

    # INSERT INTO pattern_occurrences (...)
    cur.executemany("""
    INSERT INTO pattern_occurrences (dataset_id, pattern_id, graph_id, count)
    VALUES (?, ?, ?, 1)
    """, occurrences)
    c+=1
    conn.commit()
conn.close()