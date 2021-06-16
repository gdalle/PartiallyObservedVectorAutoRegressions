import numpy as np
import pandas as pd
from pyproj import Transformer
import requests
from tqdm.notebook import tqdm

STOPS_URL = "https://opentransportdata.swiss/dataset/b9d607ba-4ff5-43a6-ac83-293f454df1fd/resource/d70319b9-3de3-4acb-9985-1bb3882b6a23/download/bav_list_current_timetable.xlsx"

STOPS_COLUMNS = {
    "Dst-Nr85": ("stop_id", "string"),
    "Ld": ("country_code", "Int64"),
    "Dst-Nr": ("stop_id_for_country", "Int64"),
    "KZ": ("control_key", "Int64"),
    "Name": ("stop_name", "string"),
    "Länge": ("stop_name_length", "Int64"),
    "Name lang": ("stop_name_long", "string"),
    "Dst-Abk": ("stop_name_short", "string"),
    "BP": ("relevant_for_transportation_plan", "category"),
    "VP": ("relevant_for_people", "category"),
    "VG": ("relevant_for_goods", "category"),
    "RB": ("relevant_for_planning_only", "category"),
    "TH": ("relevant_for_prices", "category"),
    "Status": ("status", "category"),
    "Verkehrsmittel": ("transportation_types", "category"),
    "TU-Nr": ("transport_company_id", "category"),
    "TU-Abk": ("transport_company_name", "category"),
    "GO-Nr": ("business_company_id", "category"),
    "GO-Abk": ("business_company_name", "category"),
    "Ortschaft": ("location_name", "string"),
    "Gde-Nr": ("city_id", "string"),
    "Gemeinde": ("city_name", "string"),
    "Kt.": ("district_name", "string"),
    "E-Koord.": ("lv95_e_coord", "string"),
    "N-Koord.": ("lv95_n_coord", "string"),
    "Höhe": ("altitude", "string"),
    "Bemerkungen": ("remarks", "string"),
    "Karte": ("map_hyperlink", "string"),
    "Karte.1": ("map_hyperlink_2", "string"),
}


def download_stops_data(path):
    request_result = requests.get(STOPS_URL)
    with open(path, "wb") as output_file:
        output_file.write(request_result.content)


def read_stops_data(path):
    stops = pd.read_excel(
        path,
        skiprows=[1, 2, 3],
        dtype={c: STOPS_COLUMNS[c][1] for c in STOPS_COLUMNS.keys()},
    )
    # Rename
    stops = stops.rename(
        mapper={c: STOPS_COLUMNS[c][0] for c in STOPS_COLUMNS.keys()}, axis=1
    )
    # Drop empty rows
    stops = stops[stops["stop_id"].notnull()]
    # Set index to stop_id
    stops = stops.set_index("stop_id")
    # Parse Swiss coordinates
    stops["lv95_e_coord"] = stops["lv95_e_coord"].str.replace(",", "").astype("Int64")
    stops["lv95_n_coord"] = stops["lv95_n_coord"].str.replace(",", "").astype("Int64")
    # Convert them to latitude & longitude
    transformer = Transformer.from_crs("epsg:2056", "epsg:4326")
    stops["latitude"] = np.array(
        [
            transformer.transform(stop["lv95_e_coord"], stop["lv95_n_coord"])[0]
            for _, stop in tqdm(
                stops.iterrows(), total=stops.shape[0], desc="Computing latitude "
            )
        ]
    )
    stops["longitude"] = np.array(
        [
            transformer.transform(stop["lv95_e_coord"], stop["lv95_n_coord"])[1]
            for _, stop in tqdm(
                stops.iterrows(), total=stops.shape[0], desc="Computing longitude "
            )
        ]
    )
    return stops

