import numpy as np
from src.povar import POVAR, VAR, FixedSizeSampler, Observer


def build_X(data, freq):
    """Build proxy for X based on actual train data."""
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
    """Estimate POVAR transition matrix from actual train data."""
    X, p = build_X(data, freq)
    povar = POVAR(
        var=VAR(theta=None, sigma=None),
        sampler=FixedSizeSampler(p=p),
        observer=Observer(omega=0.0),
    )
    thetas = povar.estimate_theta(
        X=X, h0=0, lambda0_range=lambda0_range, show_progress=True
    )
    return thetas
