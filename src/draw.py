import matplotlib.pyplot as plt
import networkx as nx

def get_style_for_dataset(dataset_name):
    if dataset_name.upper() == "MUTAG":
        atom_symbols = {
            "2": "C", "5": "N", "6": "O", "5": "Cl", "6": "Br"
        }
        atom_colors = {
            "2": "#909090",  # C - gris
            "5": "#3050F8",  # N - bleu
            "6": "#FF0D0D",  # O - rouge
            "5": "#00FFFF",  # Cl - cyan
            "6": "#A52A2A"   # Br - brun
        }
        bond_styles = {
            "0": "dotted",   # aromatic
            "1": "solid",    # single
            "2": "solid",    # double
            "3": "solid"     # triple
        }
        bond_widths = {
            "0": 1.5,
            "1": 2,
            "2": 3,
            "3": 4
        }
        return atom_symbols, atom_colors, bond_styles, bond_widths
    
    elif dataset_name.upper() == "PTC":

        atom_symbols = {
        "0": "In", "1": "P", "2": "O", "3": "N", "4": "Na", "5": "C", "6": "Cl", "7": "S",
        "8": "Br", "9": "F", "10": "As", "11": "K", "12": "Cu", "13": "Zn", "14": "I",
        "15": "Sn", "16": "Pb", "17": "Te", "18": "Ca"
    }

        atom_colors = {
            "0": "#C0C0C0",   # In - gris argenté
            "1": "#FFA500",   # P - orange
            "2": "#FF0D0D",   # O - rouge
            "3": "#3050F8",   # N - bleu
            "4": "#ADD8E6",   # Na - bleu clair
            "5": "#909090",   # C - gris
            "6": "#00FF00",   # Cl - vert clair
            "7": "#FFFF30",   # S - jaune
            "8": "#A52A2A",   # Br - brun
            "9": "#90E050",   # F - vert
            "10": "#FF69B4",  # As - rose vif
            "11": "#FFB6C1",  # K - rose pâle
            "14": "#940094",  # I - violet
        }

        default_atom_color = "#CCCCCC"
        bond_styles = {
        0: "solid",     # triple
        1: "solid",     # double
        2: "solid",     # single
        3: "dashed"     # aromatic
        }

        bond_widths = {
            0: 4,
            1: 3,
            2: 2,
            3: 2
        }

        return atom_symbols, atom_colors, bond_styles, bond_widths
    
    elif dataset_name.upper() == "AIDS":

        atom_symbols = {
        "0": "C", "1": "O", "2": "N", "3": "Cl", "4": "F", "5": "S", "6": "Se", "7": "P", "8": "Na",
        "9": "I", "10": "Co", "11": "Br", "12": "Li", "13": "Si", "14": "Mg", "15": "Cu", "16": "As",
        "17": "B", "18": "Pt", "19": "Ru", "20": "K", "21": "Pd", "22": "Au", "23": "Te", "24": "W",
        "25": "Rh", "26": "Zn", "27": "Bi", "28": "Pb", "29": "Ge", "30": "Sb", "31": "Sn", "32": "Ga",
        "33": "Hg", "34": "Ho", "35": "Tl", "36": "Ni", "37": "Tb"
        }

        atom_colors = {
        "0": "#909090",  # C
        "1": "#FF0D0D",  # O
        "2": "#3050F8",  # N
        "3": "#00FF00",  # Cl
        "4": "#90E050",  # F
        "5": "#FFFF30",  # S
        "6": "#FFD700",  # Se
        "7": "#FF8000",  # P
        "9": "#940094",  # I
        "11": "#A52A2A", # Br
    }

        # Autres : gris clair
        default_atom_color = "#CCCCCC"
        bond_styles = {
            "1": "solid",    # single
            "2": "solid",    # double
            "3": "solid"     # triple
        }
        bond_widths = {
            "1": 2,
            "2": 3,
            "3": 4
        }

        return atom_symbols, atom_colors, bond_styles, bond_widths
    else:
        raise NotImplementedError(f"Styles non définis pour le dataset : {dataset_name}")

def draw_graph(G: nx.Graph, dataset_name: str,figsize) -> plt.Figure:
    atom_symbols, atom_colors, bond_styles, bond_widths = get_style_for_dataset(dataset_name)

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.kamada_kawai_layout(G)

    # Nœuds
    node_labels = {n: atom_symbols.get(G.nodes[n]['label'], "?") for n in G.nodes}
    node_colors = [atom_colors.get(G.nodes[n]['label'], "#AAAAAA") for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=14, font_weight="bold", ax=ax)

    # Arêtes
    for u, v, d in G.edges(data=True):
        bond_label = str(d["label"])
        style = bond_styles.get(bond_label, "solid")
        width = bond_widths.get(bond_label, 1)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], style=style, width=width, edge_color="black", ax=ax)

    ax.axis("off")
    return fig
