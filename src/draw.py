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
    else:
        raise NotImplementedError(f"Styles non définis pour le dataset : {dataset_name}")

def draw_graph(G: nx.Graph, dataset_name: str) -> plt.Figure:
    atom_symbols, atom_colors, bond_styles, bond_widths = get_style_for_dataset(dataset_name)

    fig, ax = plt.subplots(figsize=(4, 4))
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
