import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import networkx as nx
import folium
import folium.plugins
import io
from PIL import Image
import matplotlib.pyplot as plt


def get_lims(G):
    xmin, ymin = +np.inf, +np.inf
    xmax, ymax = -np.inf, -np.inf
    for node in G.nodes():
        x, y = G.nodes[node]["longitude"], G.nodes[node]["latitude"]
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
    dx = xmax - xmin
    dy = ymax - ymin
    return (xmin - 0.05 * dx, xmax + 0.05 * dx, ymin - 0.05 * dy, ymax + 0.05 * dy)


def get_segments_and_linewidths(G, theta, weight_threshold):
    segments = []
    linewidths = []
    for edge1 in G.edges():
        for edge2 in G.edges():
            id1, id2 = G.edges[edge1]["edge_id"], G.edges[edge2]["edge_id"]
            if np.abs(theta[id1, id2]) > weight_threshold:
                x1, y1 = G.edges[edge1]["longitude"], G.edges[edge1]["latitude"]
                x2, y2 = G.edges[edge2]["longitude"], G.edges[edge2]["latitude"]
                line = [(x1, y1), (x2, y2)]
                linewidth = np.abs(theta[id1, id2])
                segments.append(line)
                linewidths.append(linewidth)
    return segments, linewidths


def map_network(G, png_path=None):
    xmin, xmax, ymin, ymax = get_lims(G)
    m = folium.Map(location=((ymin+ymax)/2, (xmin+xmax)/2), zoom_start=12.5)
    folium.plugins.Fullscreen().add_to(m)
    stations = folium.FeatureGroup(name="stations", show=False)
    edges = folium.FeatureGroup(name="edges")
    for node in G.nodes():
        attr = G.nodes[node]
        folium.Marker(
            (attr["latitude"], attr["longitude"]),
            tooltip=node,
            icon=folium.Icon(color="black"),
        ).add_to(stations)
    for edge in G.edges():
        node1, node2 = edge
        attr1 = G.nodes[node1]
        attr2 = G.nodes[node2]
        folium.PolyLine(
            locations=[
                (attr1["latitude"], attr1["longitude"]),
                (attr2["latitude"], attr2["longitude"]),
            ],
            color="black",
            weight=2,
            opacity=1,
        ).add_to(edges)
    stations.add_to(m)
    edges.add_to(m)
    folium.LayerControl().add_to(m)
    if png_path is not None:
        img_data = m._to_png(1)
        img = Image.open(io.BytesIO(img_data))
        img.save(png_path)
    return m


def plot_network(G, ax):
    edges_x = np.array(
        [
            [G.nodes[node1]["longitude"], G.nodes[node2]["longitude"]]
            for (node1, node2) in G.edges()
        ]
    ).T
    edges_y = np.array(
        [
            [G.nodes[node1]["latitude"], G.nodes[node2]["latitude"]]
            for (node1, node2) in G.edges()
        ]
    ).T
    ax.plot(edges_x, edges_y, color="black", linestyle="dotted")
    ax.scatter(
        [G.nodes[node]["longitude"] for node in G.nodes()],
        [G.nodes[node]["latitude"] for node in G.nodes()],
        marker="o",
        color="black",
    )
