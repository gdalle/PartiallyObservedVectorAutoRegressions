import pandas as pd
import numpy as np


def drop_useless_columns(data):
    data = data.drop(
        labels=[
            # we stay in Zürich
            "agency_id",
            "agency_name",
            "agency_short_name",
            # we stay on the tram
            "transportation_type",
            "transportation_subtype",
            # we already have stop id
            "stop_name_unofficial",
            # we already have line name
            "line_id",
            # we don't need this
            "circuit_transfer",
        ],
        axis=1,
    )
    return data


def uncategorize(data):
    # Uncategorize categorical variables for future group-bys
    for string_col in ["stop_id", "line_name", "trip_id"]:
        data[string_col] = data[string_col].astype("string")
    for bool_col in ["unplanned_trip", "cancelled_trip", "skipped_stop"]:
        data[bool_col] = (
            data[bool_col]
            .replace({"true": True, "false": False, np.NaN: pd.NA})
            .astype("boolean")
        )
    return data


def remove_skipped_unplanned_cancelled(data):
    perturbations = (
        (~data["skipped_stop"]) & (~data["unplanned_trip"]) & (~data["cancelled_trip"])
    )
    data = data[perturbations.values].drop(
        labels=["skipped_stop", "unplanned_trip", "cancelled_trip"], axis=1
    )
    return data


def keep_only_arrivals(data):
    departure_cols = [
        "departure_time_planned",
        "departure_time_real",
        "departure_time_status",
    ]
    data = data.drop(departure_cols, axis=1)

    # Rename arrivals as events
    data = data.dropna(subset=["arrival_time_planned"], axis=0).rename(
        mapper={
            "arrival_time_planned": "event_time_planned",
            "arrival_time_real": "event_time_real",
            "arrival_time_status": "event_time_status",
        },
        axis=1,
    )

    return data


def add_next_event(data):
    data = data.sort_values(
        ["date", "trip_id", "event_time_planned", "event_time_real"]
    )

    event_cols = [
        "stop_id",
        "event_time_planned",
        "event_time_real",
        "event_time_status",
    ]

    next_event = (
        data[event_cols].shift(-1).rename({c: "next_" + c for c in event_cols}, axis=1)
    )
    last_event_of_a_train = (
        data[["date", "trip_id"]] != data.shift(-1)[["date", "trip_id"]]
    ).any(axis=1)
    next_event.loc[last_event_of_a_train] = None

    data = pd.concat([data, next_event], axis=1)
    return data


def add_delay_columns(data):
    data["event_delay"] = (
        data["event_time_real"] - data["event_time_planned"]
    ).dt.total_seconds().astype("float32") / 60
    data["edge_duration_planned"] = (
        data["next_event_time_planned"] - data["event_time_planned"]
    ).dt.total_seconds().astype("float32") / 60
    data["edge_duration_real"] = (
        data["next_event_time_real"] - data["event_time_real"]
    ).dt.total_seconds().astype("float32") / 60
    data["edge_delay"] = (
        data["edge_duration_real"] - data["edge_duration_planned"]
    ).astype("float32")

    return data


def remove_outliers_delays(
    data,
    min_edge_duration_planned,
    max_edge_duration_planned,
    min_edge_duration_real,
    max_edge_duration_real,
    min_event_delay,
    max_event_delay,
    min_edge_delay,
    max_edge_delay,
):
    last_stop = data["next_stop_id"].isnull()
    no_event_time = data["event_time_real"].isnull()
    no_next_event_time = data["next_event_time_real"].isnull()
    short_edge_planned = data["edge_duration_planned"] < min_edge_duration_planned
    short_edge_real = data["edge_duration_real"] < min_edge_duration_real
    event_too_early = data["event_delay"] < min_event_delay
    edge_too_early = data["edge_delay"] < min_edge_delay
    long_edge_planned = data["edge_duration_planned"] > max_edge_duration_planned
    long_edge_real = data["edge_duration_real"] > max_edge_duration_real
    event_too_late = data["event_delay"] > max_event_delay
    edge_too_late = data["edge_delay"] > max_edge_delay

    dirty_data = (
        last_stop
        | no_event_time
        | no_next_event_time
        | short_edge_planned
        | long_edge_planned
        | short_edge_real
        | long_edge_real
        | event_too_early
        | event_too_late
        | edge_too_early
        | edge_too_late
    )

    return data[~dirty_data]
