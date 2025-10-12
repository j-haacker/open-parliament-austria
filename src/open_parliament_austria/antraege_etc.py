"""Module for requests to API endpoint 101

This includes:

- Antraege
- ...

The collected metadata focus on "Antraege" and associated processes.
Choices may need to change in future to support more datasets.
"""

__all__ = []

from ast import literal_eval
import numpy as np
from open_parliament_austria import (
    _download_collection_metadata,
    _download_file,
    _extract_txt_from_pdf,
    lib_data,
    _prepend_url,
)
import pandas as pd
from pathlib import Path
import pickle
import requests
import sqlite3
from typing import Literal

index_col = ["GP_CODE", "ITYP", "INR"]
raw_data = lib_data / "data" / "raw"


def append_global_metadata(global_metadata_df: pd.DataFrame):
    with sqlite3.connect(raw_data / "metadata_api_101.db") as con:
        # pickle cell contents to handle lists
        global_metadata_df.map(pickle.dumps).to_sql(
            "global", con=con, if_exists="append"
        )


def _build_global_metadataframe_from_json(
    dataset: Literal["antraege"], query_dict: dict | None = None
):
    _json = _download_collection_metadata(dataset, query_dict)
    header = pd.DataFrame.from_dict(_json["header"]).apply(
        lambda x: x == "1" if x.name.startswith("ist_") else x
    )
    global_metadata_df = pd.DataFrame(_json["rows"], columns=header.label)

    ## polish table
    date_col = [col for col in global_metadata_df.columns if "datum" in col.lower()]
    global_metadata_df[date_col] = global_metadata_df[date_col].apply(pd.to_datetime)
    global_metadata_df = global_metadata_df.apply(
        lambda col: col
        if not col.dtype == np.dtype("O")
        else col.str.replace("null", "None")
    )
    global_metadata_df = global_metadata_df.apply(
        lambda col: col
        if not col.dtype == np.dtype("O") or not all(col.str.isdecimal())
        else col.astype(int)
    )
    global_metadata_df[header[header.ist_werteliste].label] = global_metadata_df[
        header[header.ist_werteliste].label
    ].map(lambda x: x if x is None else np.array(literal_eval(x)))
    global_metadata_df = (
        global_metadata_df.drop(
            columns=[  # columns that seem irrelevant
                "ZUKZ",
                "RSS_DESC",
                "INRNUM",  # same as INR
                "LZ-Buttons",  # html button
                "sysdate???",  # download timestamp
                "WENTRY_ID",
                "NR_GP_CODE",  # same as GP_CODE
                "Gruppe",
            ]
        )
        .set_index(index_col)
        .sort_index()
    )

    append_global_metadata(global_metadata_df)


def _download_document(df_row: pd.Series):
    url = _prepend_url(df_row.HIS_URL) + "?json=True"
    _dict = requests.get(url).json()
    if "documents" in _dict["content"]:
        docs = _dict["content"]["documents"]
    if "docs" not in locals() or len(docs) == 0:
        raise Exception(f"No docs for {df_row.name}.")
    elif len(docs) > 1:
        raise Exception(f"Not implmented. Docs:\n{repr(docs)}")
    else:
        doc = docs[0]["documents"][0]
        if doc["type"] != "PDF":
            raise Exception(f"Not implemented for type {doc['type']}.")
        link = doc["link"]
        path = Path(raw_data, *list(map(str, df_row.name)))
        _download_file(_prepend_url(link), path / link.split("/")[-1])


def get_global_metadata_df(
    dataset: Literal["antraege"], query_dict: dict | None = None
):
    if not (raw_data / "metadata_api_101.db").is_file():
        _build_global_metadataframe_from_json(dataset, query_dict)
    with sqlite3.connect(raw_data / "metadata_api_101.db") as con:
        return (
            pd.read_sql("SELECT * FROM global", con=con)
            .map(lambda x: x if not isinstance(x, bytes) else pickle.loads(x))
            .set_index(index_col)
        )


def get_antragstext(df_row: pd.Series) -> str:
    path = Path(raw_data, *list(map(str, df_row.name)))
    if not path.exists():
        _download_document(df_row)
    if len(list(path.glob("*.txt"))) == 0:
        for file in path.glob("*.pdf"):
            with open(file.with_suffix(".txt"), "w") as f:
                f.write(_extract_txt_from_pdf(file))
    with open(next(path.glob("*.txt")), "r") as f:
        return f.read()
