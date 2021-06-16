from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np

from src.povar import *
from src.read_actual_data import *
from src.network import *
from src.plotting import *


def build_X(data, freq):
    minutes_since_midnight_planned = (
        data["event_time_planned"].dt.hour * 60 + data["event_time_planned"].dt.minute
    )
    minutes_since_midnight_real = (
        data["event_time_real"].dt.hour * 60 + data["event_time_real"].dt.minute
    )
    # Number of instants
    t_min = minutes_since_midnight_planned.min() // freq
    t_max = minutes_since_midnight_planned.max() // freq
    T = t_max + 1 - t_min
    # Number of days
    N = data["date"].nunique()
    # Dimension
    D = data["edge_id"].nunique()
    # Probability of observation
    p = len(data) / (N * T * D)

    # Compute time step based on actual event times
    data["time_step"] = (
        (minutes_since_midnight_real // freq - t_min).clip(0, T - 1).astype("int32")
    )
    # Average values related to the same triplet (day, time, edge)
    mean_edge_delay = data.groupby(["date", "time_step", "edge_id"])[
        "centered_edge_delay"
    ].mean()
    # Remove time step to avoid side effects
    # data = data.drop("time_step", axis=1)

    # Reindex with cartesian product [D] x [T] x [E]
    X_with_nan = mean_edge_delay.to_xarray().values
    # Fill nan values with zeros
    X = np.nan_to_num(X_with_nan)

    return X, p


def estimate_transition(data, freq, lambda0_range):
    X, p = build_X(data, freq)
    povar = POVAR(
        var=VAR(theta=None, sigma=None),
        sampler=FixedSizeSampler(p=p, replace=True, time_indep=True),
        observer=Observer(omega=0.0),
    )
    thetas = povar.estimate_theta(
        X=X, h0=0, lambda0_range=lambda0_range, show_progress=True
    )
    return thetas


def estimate_transition_and_plot(
    data, freq, log_lambda0, G, log_weight_threshold, ax
):
    theta = estimate_transition(
        data=data, freq=freq, lambda0_range=[10 ** log_lambda0],
    )

    ax.clear()
    plot_network(G, ax)
    segments, linewidths = get_segments_and_linewidths(
        G, theta, 10 ** log_weight_threshold
    )
    xmin, xmax, ymin, ymax = get_lims(G)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("longitude")
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("latitude")
    lc = LineCollection(
        segments=segments, linewidths=linewidths, label="transition weights",
    )
    ax.add_collection(lc)
    ax.set_aspect("equal")


def get_distance(G, edge1, edge2):
    euclidean_distance = np.sqrt(
        np.square(G.edges[edge1]["latitude"] - G.edges[edge2]["latitude"])
        + np.square(G.edges[edge1]["longitude"] - G.edges[edge2]["longitude"])
    )
    s1, t1 = edge1
    s2, t2 = edge2
    path_distance = min(
        G.nodes[s1]["distance"].get(s2, np.inf),
        G.nodes[s2]["distance"].get(s1, np.inf),
    )
    return path_distance


def get_distance_matrix(G):
    E = G.number_of_edges()
    d = np.empty((E, E))
    for edge1 in G.edges():
        for edge2 in G.edges():
            id1, id2 = G.edges[edge1]["edge_id"], G.edges[edge2]["edge_id"]
            distance = get_distance(G, edge1, edge2)
            d[id1, id2] = distance
    return d
