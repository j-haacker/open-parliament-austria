"""Module for requests to API endpoint 211

This includes:

- Steno-protokolle
- ...

The collected metadata focus on "Protokolle" and associated processes.
Choices may need to change in future to support more datasets.
"""

__all__ = []

# from aiohttp import ClientSession
# import asyncio
# from bs4 import BeautifulSoup
# from collections.abc import Iterable
# from contextlib import contextmanager
import numpy as np
from open_parliament_austria import (
    _add_missing_db_cols,
    _append_global_metadata,
    _download_collection_metadata,
    #     _download_file,
    _ensure_allowed_sql_name,
    #     _extract_txt_from_pdf,
    #     _get_rowid_index,
    _get_colnames,
    #     _get_colname_by_type,
    _get_db_connector,
    _get_pd_sql_reader,
    #     _prepend_url,
    #     _sqlit/e3_type,
)
from open_parliament_austria.resources import _column_name_dict_211
import pandas as pd

# from pathlib import Path
# import pickle
# import requests
import sqlite3
# from typing import Any, Literal

sqlite3.register_adapter(np.int_, lambda i: int(i))

db_con = _get_db_connector("metadata_api_211.db")
index_col = ["Periode", "Gremium", "Nummer"]
pd_read_sql = _get_pd_sql_reader(index_col)


def download_global_metadata(query_dict: dict | None = None):
    _json = _download_collection_metadata(dataset="sitzungen", query_dict=query_dict)
    # header = pd.DataFrame.from_dict(_json["header"])
    # header.to_csv("tmp_header.csv")
    # pd.DataFrame.from_dict(_json["rows"]).to_csv("tmp_rows.csv")
    # print(header.to_string())
    header = (
        pd.DataFrame.from_dict(_json["header"])
        .apply(lambda x: x == "1" if x.name.startswith("ist_") else x)
        .iloc[: len(_json["rows"][0])]
        .set_index("feldId")
    )
    global_metadata_df = pd.DataFrame(_json["rows"], columns=header.index)
    global_metadata_df.drop(
        columns=[
            col
            for col in global_metadata_df.columns
            if col not in _column_name_dict_211
        ],
        inplace=True,
    )
    global_metadata_df.rename(columns=_column_name_dict_211, inplace=True)
    global_metadata_df.dropna(axis=1, how="all", inplace=True)

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
    # list_col = [
    #     _column_name_dict_211[idx] for idx, val in header.ist_werteliste.items() if val
    # ]
    # global_metadata_df[list_col] = global_metadata_df[list_col].map(
    #     lambda x: x if x is None else np.array(literal_eval(x))
    # )
    global_metadata_df.set_index(index_col, inplace=True)
    global_metadata_df.sort_index(inplace=True)
    print(global_metadata_df.to_string())

    ## write to db
    with db_con() as con:
        print(_get_colnames(con, "global"))
    _create_global_db_tbl()
    with db_con() as con:
        _add_missing_db_cols(con, "global", global_metadata_df)
        _append_global_metadata(con, global_metadata_df)


def _create_global_db_tbl():
    with db_con() as con:
        sql = (
            "CREATE TABLE IF NOT EXISTS global({0} TEXT, {1} TEXT, {2} INTEGER); "
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_global_{0}_{1}_{2} ON global({0}, {1}, {2});"
        )
        print(sql.format(*index_col))
        con.executescript(sql.format(*index_col))


def _create_child_db_tbl(
    table_name: str, columns: list[str] = [], _types: list[str] = []
):
    for name in [table_name, *columns, *_types]:
        _ensure_allowed_sql_name(name)
    sql = (
        "CREATE TABLE IF NOT EXISTS {3}("
        "{0} TEXT, {1} TEXT, {2} DATETIME, "
        + ", ".join([f"{c} {t}" for c, t in zip(columns, _types)])
        + "FOREIGN KEY ({0}, {1}, {2}) REFERENCES global({0}, {1}, {2})"
        ")"
    ).format(*index_col, table_name)
    with db_con() as con:
        con.execute(sql)


# def get_geschichtsseiten(index: Iterable[tuple[str, str, int]]) -> pd.DataFrame:
#     if not db_path().is_file():
#         raise Exception(
#             "Error: Database not found. Initialize database using "
#             "`get_global_metadata_df(dataset, query)`."
#         )
#     with db_con() as con:
#         if (
#             con.execute(
#                 "SELECT 1 FROM sqlite_master WHERE name='geschichtsseiten'"
#             ).fetchone()
#             is None
#         ):
#             _create_child_db_tbl("geschichtsseiten")
#         for idx in index:  # TODO add concurrent download?
#             if (
#                 con.execute(
#                     "SELECT 1 FROM geschichtsseiten "
#                     "WHERE "
#                     + " AND ".join(
#                         [
#                             f"{col} = {_quote_if_str(val)}"
#                             for col, val in zip(index_col, idx)
#                         ]
#                     )
#                 ).fetchone()
#                 is None
#             ):
#                 _dict = requests.get(
#                     _prepend_url(_query_single_value("HIS_URL", idx, "global", con)),
#                     {"json": "True"},
#                 ).json()["content"]
#                 [
#                     _dict.pop(k)
#                     for k in [
#                         *[
#                             ix.lower()
#                             for ix in _get_colnames(con, "global")
#                             if isinstance(ix, str)
#                         ],
#                         "breadcrumbs",
#                         "type",
#                         "sntype",
#                         "title",
#                         "nr_gp_code",
#                         "intranet",
#                         "description",
#                         "headwords",
#                         "topics",
#                         "names",
#                     ]
#                     if k in _dict
#                 ]
#                 new_row = pd.DataFrame.from_records(
#                     [{k: v for k, v in _dict.items()}],
#                     index=pd.MultiIndex.from_tuples((idx,), names=index_col),
#                 )
#                 new_row.rename(
#                     columns={"update": "_update", "group": "_group"}, inplace=True
#                 )
#                 new_row[["_update", "einlangen"]] = new_row[
#                     ["_update", "einlangen"]
#                 ].transform(pd.to_datetime)
#                 dtype_dict = {k: _sqlite3_type(v) for k, v in new_row.iloc[0].items()}
#                 db_col_set = set(_get_colnames(con, "geschichtsseiten"))
#                 missing_cols = [
#                     col
#                     for col in db_col_set
#                     if col not in list(new_row.columns) + index_col
#                 ]
#                 new_row[missing_cols] = [None] * len(missing_cols)
#                 dtype_dict.update(
#                     {k: "NUMERIC" for k in missing_cols}
#                 )  # type not known
#                 for col in [col for col in new_row.columns if col not in db_col_set]:
#                     _ensure_allowed_col_name(col)
#                     con.execute(
#                         f"ALTER TABLE geschichtsseiten ADD COLUMN {col} {dtype_dict[col]}"
#                     )
#                 new_row.transform(
#                     lambda col: col
#                     if dtype_dict[col.name] != "BLOB"
#                     else col.map(pickle.dumps)
#                 ).to_sql("geschichtsseiten", con, if_exists="append", dtype=dtype_dict)
#         return pd_read_sql(con, "geschichtsseiten", index=index)


def get_global_metadata_df() -> pd.DataFrame:
    try:
        with db_con() as con:
            return pd_read_sql(con, "global")
    except FileNotFoundError as err:
        raise Exception(
            "FileNotFoundError: Likely database is missing. If you haven't, run "
            f"`download_global_metadata(query_dict)`. Original error was:\n{str(err)}"
        )
