import networkx as nx

def parse_gspan_file(path):
    graphs = []
    current_graph = None
    current_id = None

    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue

            if tokens[0] == 't':
                if current_graph is not None:
                    graphs.append((current_id, current_graph))
                current_id = int(tokens[2])
                current_graph = nx.Graph()
            elif tokens[0] == 'v':
                node_id = int(tokens[1])
                label = tokens[2]
                current_graph.add_node(node_id, label=label)
            elif tokens[0] == 'e':
                source = int(tokens[1])
                target = int(tokens[2])
                edge_label = tokens[3]
                current_graph.add_edge(source, target, label=edge_label)

        # Ajouter le dernier graphe
        if current_graph is not None:
            graphs.append((current_id, current_graph))

    return graphs