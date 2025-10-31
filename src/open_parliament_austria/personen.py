"""Module for requests to API endpoint 409

This includes:

- Parlamentarier*innen
- ...

The collected metadata focus on those and associated processes.
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
    _ensure_allowed_sql_name,
    #     _extract_txt_from_pdf,
    #     _get_rowid_index,
    _get_coll_downloader,
    _get_db_connector,
    _get_pd_sql_reader,
    #     _prepend_url,
    #     _sqlit/e3_type,
)
import pandas as pd

# from pathlib import Path
# import pickle
# import requests
import sqlite3
# from typing import Any, Literal

sqlite3.register_adapter(np.int_, lambda i: int(i))

db_con = _get_db_connector("metadata_api_409.db")
index_col = ["Personen_ID"]
index_sqltypes = ["INTEGER"]
pd_read_sql = _get_pd_sql_reader(index_col)
download_global_metadata = _get_coll_downloader(
    db_con, "personen", index_col, index_sqltypes
)


def _create_child_db_tbl(
    table_name: str, columns: list[str] = [], _types: list[str] = []
):
    for name in [table_name, *columns, *_types]:
        _ensure_allowed_sql_name(name)
    sql = (
        "CREATE TABLE IF NOT EXISTS {1}("
        "{0} INTEGER, "
        + ", ".join([f"{c} {t}" for c, t in zip(columns, _types)])
        + "FOREIGN KEY {0} REFERENCES global({0})"
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
