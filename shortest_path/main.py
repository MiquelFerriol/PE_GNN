import ignnition

import os
import networkx as nx
import matplotlib.pyplot as plt
import json
from networkx.readwrite import json_graph
import ignnition
import numpy as np
import random

def generate_random_graph(min_nodes, max_nodes, min_edge_weight, max_edge_weight, p):
    while True:
        # Create a random Erdos Renyi graph
        G = nx.erdos_renyi_graph(random.randint(min_nodes, max_nodes), p)
        complement = list(nx.k_edge_augmentation(G, k=1, partial=True))
        G.add_edges_from(complement)
        nx.set_node_attributes(G, 0, 'src_tgt')
        nx.set_node_attributes(G, 0, 'sp')
        nx.set_node_attributes(G, 'node', 'entity')

        # Assign randomly weights to graph edges
        for (u, v, w) in G.edges(data=True):
            w['weight'] = random.randint(min_edge_weight, max_edge_weight)

        # Select the source and target nodes to compute the shortest path
        src, tgt = random.sample(list(G.nodes), 2)

        G.nodes[src]['src_tgt'] = 1
        G.nodes[tgt]['src_tgt'] = 1

        # Compute all the shortest paths between source and target nodes
        try:
            shortest_paths = list(nx.all_shortest_paths(G, source=src, target=tgt,weight='weight'))
        except:
            shortest_paths = []
        # Check if there exists only one shortest path
        if len(shortest_paths) == 1:
            if len(shortest_paths[0])>=3 and len(shortest_paths[0])<=5:
                for node in shortest_paths[0]:
                    G.nodes[node]['sp'] = 1
                return shortest_paths[0], nx.DiGraph(G)

def print_graph_predictions(G, path, predictions,ax):
    predictions = np.array(predictions)
    node_border_colors = []
    links = []
    for i in range(len(path)-1):
        links.append([path[i], path[i+1]])
        links.append([path[i+1], path[i]])

    # Add colors to node borders for source and target nodes
    for node in G.nodes(data=True):
        if node[1]['src_tgt'] == 1:
            node_border_colors.append('red')
        else:
            node_border_colors.append('white')
    # Add colors for predictions [0,1]
    node_colors = predictions

    # Add colors for edges
    edge_colors = []
    for edge in G.edges(data=True):
        e=[edge[0],edge[1]]
        if e in links:
            edge_colors.append('red')
        else:
            edge_colors.append('black')
    pos= nx.shell_layout(G)
    vmin = node_colors.min()
    vmax = node_colors.max()
    vmin = 0
    vmax = 1
    cmap = plt.cm.coolwarm
    nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors, cmap=cmap, vmin=vmin, vmax=vmax,
                           edgecolors=node_border_colors, linewidths=4, ax=ax)
    nx.draw_networkx_edges(G, pos=pos, edge_color=edge_colors, arrows=False, ax=ax, width=2)
    nx.draw_networkx_edge_labels(G, pos=pos, label_pos=0.5, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax)

def print_graph_solution(G, path, predictions, ax, pred_th):
    predictions = np.array(predictions)
    node_colors = []
    node_border_colors = []
    links = []
    for i in range(len(path)-1):
        links.append([path[i], path[i+1]])
        links.append([path[i+1], path[i]])

    # Add colors on node borders for source and target nodes
    for node in G.nodes(data=True):
        if node[1]['src_tgt'] == 1:
            node_border_colors.append('red')
        else:
            node_border_colors.append('white')

    # Add colors for predictions Blue or Red
    cmap = plt.cm.get_cmap('coolwarm')
    dark_red = cmap(1.0)
    for p in predictions:
        if p >= pred_th:
            node_colors.append(dark_red)
        else:
            node_colors.append('blue')

    # Add colors for edges
    edge_colors = []
    for edge in G.edges(data=True):
        e=[edge[0],edge[1]]
        if e in links:
            edge_colors.append('red')
        else:
            edge_colors.append('black')
    pos= nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors, edgecolors=node_border_colors, linewidths=4, ax=ax)
    nx.draw_networkx_edges(G, pos=pos, edge_color=edge_colors, arrows=False, ax=ax, width=2)
    nx.draw_networkx_edge_labels(G, pos=pos, label_pos=0.5, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax)

def print_input_graph(G, sh_path, ax):
    node_colors = []
    node_border_colors = []

    # Add colors to node borders for source and target nodes
    for node in G.nodes(data=True):
        if node[1]['src_tgt'] == 1:
            node_border_colors.append('red')
        else:
            node_border_colors.append('white')

    links = []
    for i in range(len(sh_path)-1):
        links.append([sh_path[i], sh_path[i+1]])
        links.append([sh_path[i+1], sh_path[i]])

    edge_colors = []
    for edge in G.edges(data=True):
        e=[edge[0],edge[1]]
        if e in links:
            edge_colors.append('red')
        else:
            edge_colors.append('black')
    pos= nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos=pos, edgecolors=node_border_colors, linewidths=4, ax=ax)
    nx.draw_networkx_edges(G, pos=pos, edge_color=edge_colors, arrows=False, ax=ax, width=2)
    nx.draw_networkx_edge_labels(G, pos=pos, label_pos=0.5, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=ax)

model = ignnition.create_model(model_dir= './')
model.computational_graph()
model.train_and_validate()

dataset_samples = []
sh_path, G = generate_random_graph(min_nodes=8, max_nodes=12, min_edge_weight=1, max_edge_weight=10, p=0.3)
graph = G.to_undirected()
dataset_samples.append(json_graph.node_link_data(G))

# write prediction dataset
root_dir="./data"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
if not os.path.exists(root_dir+"/test"):
    os.makedirs(root_dir+"/test")
with open(root_dir+"/test/data.json", "w") as f:
    json.dump(dataset_samples, f)

# Make predictions
predictions = model.predict()

# Print the results
fig, axes = plt.subplots(nrows=1, ncols=3)
ax = axes.flatten()

# Print input graph
ax1 = ax[0]
ax1.set_title("Input graph")
print_input_graph(graph, sh_path, ax1)

# Print graph with predictions (soft values)
ax1 = ax[1]
ax1.set_title("GNN predictions (soft values)")
print_graph_predictions(graph, sh_path, predictions[0], ax1)

# Print solution of the GNN
pred_th = 0.5
ax1 = ax[2]
ax1.set_title("GNN solution (p >= "+str(pred_th)+")")
print_graph_solution(graph, sh_path, predictions[0], ax1, pred_th)
print("True path:", sh_path)
# Show plot in full screen
plt.rcParams['figure.figsize'] = [10, 4]
plt.rcParams['figure.dpi'] = 100
plt.tight_layout()
plt.show()