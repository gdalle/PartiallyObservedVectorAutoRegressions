import pandas as pd
import networkx as nx
import numpy as np


def build_network(data, stops, max_edge_rank):
    """Build tramway network from most frequent edges."""
    # Create graph object
    G = nx.DiGraph()
    # Add nodes
    for stop_id in np.union1d(data["stop_id"].unique(), data["next_stop_id"].unique()):
        G.add_node(stop_id)
        if stop_id in stops.index:
            G.nodes[stop_id]["stop_name"] = stops.loc[stop_id, "stop_name"]
            G.nodes[stop_id]["latitude"] = stops.loc[stop_id, "latitude"]
            G.nodes[stop_id]["longitude"] = stops.loc[stop_id, "longitude"]
        else:
            G.nodes[stop_id]["stop_name"] = None
            G.nodes[stop_id]["latitude"] = np.nan
            G.nodes[stop_id]["longitude"] = np.nan

    # Compute edge information to store it in the graph
    edge_groupby = data.groupby(["stop_id", "next_stop_id"])
    edge_df = pd.DataFrame(index=edge_groupby.size().index)
    edge_df["count"] = edge_groupby.size().values
    edge_df["rank"] = edge_df["count"].rank(ascending=False).astype(int)
    edge_df["duration"] = edge_groupby["edge_duration_planned"].median()

    # Add edges
    for k, (stop_id, next_stop_id) in enumerate(edge_df.index):
        G.add_edge(
            stop_id,
            next_stop_id,
            edge_rank=int(edge_df.loc[(stop_id, next_stop_id), "rank"]),
            duration=edge_df.loc[(stop_id, next_stop_id), "duration"],
        )
        G.edges[stop_id, next_stop_id]["latitude"] = (
            0.5 * G.nodes[stop_id]["latitude"] + 0.5 * G.nodes[next_stop_id]["latitude"]
        )
        G.edges[stop_id, next_stop_id]["longitude"] = (
            0.5 * G.nodes[stop_id]["longitude"]
            + 0.5 * G.nodes[next_stop_id]["longitude"]
        )

    # Keep most frequent edges
    frequent_edges = edge_df[edge_df["rank"] <= max_edge_rank].index.tolist()
    G = G.edge_subgraph(frequent_edges)

    # Keep largest connected component
    connected_components = list(nx.strongly_connected_components(G))
    index_largest_connected_component = np.argmax(list(map(len, connected_components)))
    largest_connected_component = connected_components[
        index_largest_connected_component
    ]
    G = G.subgraph(largest_connected_component)

    # Compute shortest paths between nodes
    for (node, (distance, path)) in nx.all_pairs_dijkstra(G, weight="duration"):
        G.nodes[node]["distance"] = distance

    return G


def get_distance(G, edge1, edge2):
    """Retrieve pre-computed inter-edge distance."""
    s1, _ = edge1
    s2, _ = edge2
    path_distance = min(
        G.nodes[s1]["distance"].get(s2, np.inf),
        G.nodes[s2]["distance"].get(s1, np.inf),
    )
    return path_distance


def get_distance_matrix(G):
    """Build matrix for faster access to inter-edge distances."""
    E = G.number_of_edges()
    d = np.empty((E, E))
    for edge1 in G.edges():
        for edge2 in G.edges():
            id1, id2 = G.edges[edge1]["edge_id"], G.edges[edge2]["edge_id"]
            distance = get_distance(G, edge1, edge2)
            d[id1, id2] = distance
    return d
