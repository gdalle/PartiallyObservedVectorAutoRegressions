import itertools
import os
import zipfile
from joblib import Parallel, delayed
import pandas as pd
from tqdm.notebook import tqdm

# Constants

YEARS = ["2018", "2019", "2020", "2021"]
MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

DATA_COLUMNS = {
    "BETRIEBSTAG": ("date", "string"),
    "FAHRT_BEZEICHNER": ("trip_id", "category"),
    "BETREIBER_ID": ("agency_id", "category"),
    "BETREIBER_ABK": ("agency_short_name", "category"),
    "BETREIBER_NAME": ("agency_name", "category"),
    "PRODUKT_ID": ("transportation_type", "category"),
    "LINIEN_ID": ("line_id", "category"),
    "LINIEN_TEXT": ("line_name", "category"),
    "UMLAUF_ID": ("circuit_transfer", "category"),
    "VERKEHRSMITTEL_TEXT": ("transportation_subtype", "category"),
    "ZUSATZFAHRT_TF": ("unplanned_trip", "category"),
    "FAELLT_AUS_TF": ("cancelled_trip", "category"),
    "BPUIC": ("stop_id", "category"),
    "HALTESTELLEN_NAME": ("stop_name_unofficial", "category"),
    "ANKUNFTSZEIT": ("arrival_time_planned", "string"),
    "AN_PROGNOSE": ("arrival_time_real", "string"),
    "AN_PROGNOSE_STATUS": ("arrival_time_status", "category"),
    "ABFAHRTSZEIT": ("departure_time_planned", "string"),
    "AB_PROGNOSE": ("departure_time_real", "string"),
    "AB_PROGNOSE_STATUS": ("departure_time_status", "category"),
    "DURCHFAHRT_TF": ("skipped_stop", "category"),
}

AGENCY_NAMES = ["Verkehrsbetriebe Zürich", "Verkehrsbetriebe Zürich INFO+"]
TRANSPORTATION_TYPES = ["Tram"]

# Utils


def concat_preserving_categorical(dfs):
    """Concatenate while preserving categorical columns."""
    columns, dtypes = dfs[0].columns, dfs[0].dtypes
    res = pd.DataFrame()
    for c in tqdm(columns, desc="Concatenation "):
        if str(dtypes[c]) == "category":
            res[c] = pd.api.types.union_categoricals(
                [df[c].astype("category") for df in dfs]
            )
        else:
            res[c] = pd.concat([df[c] for df in dfs])
    return res


# Read CSV files


def read_day_csv(daily_csv_path):
    """Read daily csv in the right format."""
    try:
        data = pd.read_csv(
            daily_csv_path,
            sep=";",
            dtype={c: DATA_COLUMNS[c][1] for c in DATA_COLUMNS.keys()},
        )
    except UnicodeDecodeError:
        print("Skipped (UTF-8 error): ", daily_csv_path)
        return None
    # Rename columns
    data = data.rename(
        mapper={c: DATA_COLUMNS[c][0] for c in DATA_COLUMNS.keys()}, axis=1
    )
    # Convert datetime columns
    for timecol in ["date"]:
        data[timecol] = pd.to_datetime(
            data[timecol], format="%d.%m.%Y", errors="coerce"
        )
    for timecol in ["arrival_time_planned", "departure_time_planned"]:
        data[timecol] = pd.to_datetime(
            data[timecol], format="%d.%m.%Y %H:%M", errors="coerce"
        )
    for timecol in ["arrival_time_real", "departure_time_real"]:
        data[timecol] = pd.to_datetime(
            data[timecol], format="%d.%m.%Y %H:%M:%S", errors="coerce"
        )
    # Translate columns in German
    for status_col in ["arrival_time_status", "departure_time_status"]:
        data[status_col] = (
            data[status_col]
            .replace(
                {
                    "PROGNOSE": "Forecast",
                    "GESCHAETZT": "Estimated",
                    "UNBEKANNT": "Unknown",
                    "REAL": "Real",
                }
            )
            .fillna("Forecast")
            .astype("category")
        )
    data["transportation_type"] = (
        data["transportation_type"]
        .replace(
            {
                "Zug": "Train",
                "Bus": "Bus",
                "BUS": "Bus",
                "Schiff": "Boat",
                "Tram": "Tram",
            }
        )
        .fillna("Unknown")
        .astype("category")
    )
    return data


# A pyramid of decompression and recompression


def unzip_single_month_store_days(
    monthly_zip_dir_path, monthly_zip_name, daily_parquet_dir_path
):
    """Read a single zipped month full of csv and split it into parquet days."""
    monthly_zip_path = os.path.join(monthly_zip_dir_path, monthly_zip_name)
    with zipfile.ZipFile(monthly_zip_path) as monthly_zip_file:
        # Loop over all days of the month
        for daily_csv_name in monthly_zip_file.namelist():
            # Skip additional files
            if ".csv" not in daily_csv_name:
                continue
            # Open and parse csv
            with monthly_zip_file.open(daily_csv_name, "r") as daily_csv_file:
                daily_data = read_day_csv(daily_csv_file)
            # Filter
            if daily_data is not None:
                daily_data = daily_data[
                    daily_data["agency_name"].isin(AGENCY_NAMES)
                    & daily_data["transportation_type"].isin(TRANSPORTATION_TYPES)
                ]
                # Save as parquet file
                parquet_file_name = daily_csv_name.split("/")[-1][:10] + ".parquet"
                daily_data.to_parquet(
                    os.path.join(daily_parquet_dir_path, parquet_file_name)
                )


def unzip_months_store_days(
    monthly_zip_dir_path,
    daily_parquet_dir_path,
):
    """Read all zipped months full of csv and split them into parquet days."""
    monthly_zip_names = [
        name
        for name in sorted(os.listdir(monthly_zip_dir_path))
        if name.startswith("19_") or name.startswith("18_")
    ]
    Parallel(n_jobs=6)(
        delayed(unzip_single_month_store_days)(
            monthly_zip_dir_path, monthly_zip_name, daily_parquet_dir_path
        )
        for monthly_zip_name in tqdm(
            monthly_zip_names, desc="Decompressing months for 2018-2019"
        )
    )


def read_days_store_months(daily_parquet_dir_path, monthly_parquet_dir_path):
    """Read parquet days and put them together into months."""
    for (year, month) in list(itertools.product(YEARS, MONTHS)):
        yearmonth = "{}-{}".format(year, month)
        daily_files = sorted(
            [file for file in os.listdir(daily_parquet_dir_path) if yearmonth in file]
        )
        if daily_files:
            monthly_data_list = []
            for date in tqdm(daily_files, desc="Reading " + yearmonth):
                daily_data = pd.read_parquet(os.path.join(daily_parquet_dir_path, date))
                monthly_data_list.append(daily_data)
            monthly_data = concat_preserving_categorical(monthly_data_list)
            monthly_data.to_parquet(
                os.path.join(monthly_parquet_dir_path, yearmonth + ".parquet")
            )


def read_months_return_full(
    monthly_parquet_dir_path, years=[2018, 2019]
):
    """Read parquet months and put them together into a full dataframe."""
    data = concat_preserving_categorical(
        [
            pd.read_parquet(os.path.join(monthly_parquet_dir_path, monthly_file))
            for monthly_file in tqdm(
                sorted(os.listdir(monthly_parquet_dir_path)), desc="Reading files "
            )
            if any(str(year) in monthly_file for year in years)
        ]
    )
    return data
