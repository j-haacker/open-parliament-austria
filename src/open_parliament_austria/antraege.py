"""Module for requests to API endpoint 101

This includes:

- Antraege
- ...

The collected metadata focus on "Antraege" and associated processes.
Choices may need to change in future to support more datasets.
"""

__all__ = []

from aiohttp import ClientSession
import asyncio
from bs4 import BeautifulSoup
from collections.abc import Iterable
import numpy as np
from open_parliament_austria import (
    _download_file,
    _ensure_allowed_sql_name,
    _extract_txt_from_pdf,
    _get_coll_downloader,
    _get_colnames,
    _get_db_connector,
    _get_pd_sql_reader,
    _prepend_url,
    _quote_if_str,
    raw_data,
    _sqlite3_type,
)
import pandas as pd
from pathlib import Path
import pickle
import requests
import sqlite3
from typing import Any, Literal

sqlite3.register_adapter(np.int_, lambda i: int(i))

db_con = _get_db_connector("metadata_api_101.db")
index_col = ["GP_CODE", "ITYP", "INR"]
index_sqltypes = ["TEXT", "TEXT", "INTEGER"]
pd_read_sql = _get_pd_sql_reader(index_col)
download_global_metadata = _get_coll_downloader(
    db_con, "antraege", index_col, index_sqltypes
)


def append_global_metadata(global_metadata_df: pd.DataFrame):
    dtype_dict = {
        k: _sqlite3_type(v)
        for k, v in global_metadata_df.dropna(axis=0).iloc[0].items()
    }
    with db_con() as con:
        global_metadata_df.transform(
            lambda col: col if dtype_dict[col.name] != "BLOB" else col.map(pickle.dumps)
        ).to_sql("global", con=con, if_exists="append", dtype=dtype_dict)


def _create_child_db_tbl(
    table_name: str, columns: list[str] = [], _types: list[str] = []
):
    for name in [table_name, *columns, *_types]:
        _ensure_allowed_sql_name(name)
    sql = (
        "CREATE TABLE IF NOT EXISTS {3}("
        "{0} TEXT, {1} TEXT, {2} INTEGER, "
        + ", ".join([f"{c} {t}" for c, t in zip(columns, _types)])
        + "FOREIGN KEY ({0}, {1}, {2}) REFERENCES global({0}, {1}, {2})"
        ")"
    ).format(*index_col, table_name)
    with db_con() as con:
        con.execute(sql)


def _create_raw_text_db_tbl():
    with db_con() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS raw_text("
            "{0} TEXT, {1} TEXT, {2} INTEGER, "
            "file_name TEXT, title TEXT, raw_text TEXT, "
            "FOREIGN KEY ({0}, {1}, {2}) REFERENCES global({0}, {1}, {2})"
            ")".format(*index_col)
        )


# def _download_document(idx: tuple[str, str, int]):
#     [d["documents"] for d in _query_single_value("documents", idx, "geschichtsseiten")]
#     if "docs" not in locals() or len(docs) == 0:  # noqa: F821
#         raise Exception(f"No docs for {idx}.")
#     elif len(docs) > 2:  # noqa: F821
#         # See what docs are included and implement handling
#         raise Exception(f"Not implmented. Docs:\n{repr(docs)}")  # noqa: F821
#     elif len(docs) > 1:  # noqa: F821
#         # prefer digital version
#         if any(["(elektr. übermittelte Version)" in doc["title"] for doc in docs]):  # noqa: F821
#             docs = next(
#                 doc["documents"]
#                 for doc in docs  # noqa: F821
#                 if "(elektr. übermittelte Version)" in doc["title"]
#             )
#         # otherwise select original
#         elif any(["(gescanntes Original)" in doc["title"] for doc in docs]):
#             docs = next(
#                 doc["documents"]
#                 for doc in docs
#                 if "(gescanntes Original)" in doc["title"]
#             )
#         else:
#             docs = docs[0]["documents"]
#     if any(["HTML" in d["type"] for d in docs]):
#         link = next(d["link"] for d in docs if "HTML" in d["type"])
#     else:
#         doc = docs[0]
#         if docs["type"] != "PDF":
#             raise Exception(f"Not implemented for type {doc['type']}.")
#         link = doc["link"]
#     path = Path(raw_data(), *list(map(str, idx)))
#     asyncio.run(
#         _download_file(ClientSession(), _prepend_url(link), path / link.split("/")[-1])
#     )


def _query_single_value(
    col: str,
    idx: tuple[str, str, int],
    tbl: Literal["global", "geschichtsseiten", "raw_text"] = "global",
    con: sqlite3.Connection | None = None,
) -> Any:
    # print(idx)
    if tbl not in ["global", "geschichtsseiten", "raw_text"]:
        raise Exception(f"Table name {tbl} not allowed.")

    def _inner(con: sqlite3.Connection):
        if col not in _get_colnames(con, tbl):
            raise Exception(f"Column name {col} not in {_get_colnames(con, tbl)}.")
        return con.execute(
            f"SELECT {col} FROM {tbl} WHERE GP_CODE = ? AND ITYP = ? AND INR = ?", idx
        ).fetchone()[0]

    if con:
        return _inner(con)
    with db_con() as con:
        return _inner(con)


def get_antragstext(idx: tuple[str, str, int], file_name: str) -> str:
    query = "SELECT raw_text FROM raw_text WHERE" + 4 * " {} = ? AND"

    def _fetch():
        return con.execute(
            query[:-4].format(*index_col, "file_name"), [*idx, file_name]
        ).fetchone()

    with db_con() as con:
        try:
            text = _fetch()[0]
        except (sqlite3.OperationalError, TypeError) as err:
            if str(err).startswith("no such table"):
                _create_raw_text_db_tbl()
            elif str(err) != "'NoneType' object is not subscriptable":
                raise err
            path = Path(raw_data(), *list(map(str, idx)), file_name)
            if not path.exists():
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                task = loop.create_task(
                    _download_file(
                        ClientSession(loop=loop),
                        _prepend_url(
                            "/dokument/" + "/".join(list(map(str, idx)) + [file_name])
                        ),
                        path,
                    )
                )
                loop.run_until_complete(task)
            if path.suffix.lower() == ".pdf":
                text = _extract_txt_from_pdf(path)
            elif path.suffix.lower() in [".html", ".hml"]:
                with open(path, "r") as f_in:
                    text = BeautifulSoup(f_in.read(), "html.parser").body.text
            else:
                raise Exception(f"File extension not recognized: {file_name}")
            for doc in pickle.loads(
                _query_single_value("documents", idx, "geschichtsseiten", con)
            ):
                if any([d["link"].endswith(file_name) for d in doc["documents"]]):
                    break
            title = doc["title"]
            con.execute(
                "INSERT INTO raw_text VALUES(" + 5 * "?, " + "?)",
                [*idx, file_name, title, text],
            )
    return text


def get_geschichtsseiten(index: Iterable[tuple[str, str, int]]) -> pd.DataFrame:
    with db_con() as con:
        db_tables = [n[0] for n in con.execute("SELECT name FROM sqlite_master")]
        if "global" not in db_tables:
            raise Exception(
                "Error: Database not found. Initialize database using "
                "`get_global_metadata_df(dataset, query)`."
            )
        elif "geschichtsseiten" not in db_tables:
            _create_child_db_tbl("geschichtsseiten")
        for idx in index:  # TODO add concurrent download?
            if (
                con.execute(
                    "SELECT 1 FROM geschichtsseiten "
                    "WHERE "
                    + " AND ".join(
                        [
                            f"{col} = {_quote_if_str(val)}"
                            for col, val in zip(index_col, idx)
                        ]
                    )
                ).fetchone()
                is None
            ):
                _dict = requests.get(
                    _prepend_url(_query_single_value("HIS_URL", idx, "global", con)),
                    {"json": "True"},
                ).json()["content"]
                [
                    _dict.pop(k)
                    for k in [
                        *[
                            ix.lower()
                            for ix in _get_colnames(con, "global")
                            if isinstance(ix, str)
                        ],
                        "breadcrumbs",
                        "type",
                        "sntype",
                        "title",
                        "nr_gp_code",
                        "intranet",
                        "description",
                        "headwords",
                        "topics",
                        "names",
                    ]
                    if k in _dict
                ]
                new_row = pd.DataFrame.from_records(
                    [{k: v for k, v in _dict.items()}],
                    index=pd.MultiIndex.from_tuples((idx,), names=index_col),
                )
                new_row.rename(
                    columns={"update": "_update", "group": "_group"}, inplace=True
                )
                new_row[["_update", "einlangen"]] = new_row[
                    ["_update", "einlangen"]
                ].transform(pd.to_datetime)
                dtype_dict = {k: _sqlite3_type(v) for k, v in new_row.iloc[0].items()}
                db_col_set = set(_get_colnames(con, "geschichtsseiten"))
                missing_cols = [
                    col
                    for col in db_col_set
                    if col not in list(new_row.columns) + index_col
                ]
                new_row[missing_cols] = [None] * len(missing_cols)
                dtype_dict.update(
                    {k: "NUMERIC" for k in missing_cols}
                )  # type not known
                for col in [col for col in new_row.columns if col not in db_col_set]:
                    _ensure_allowed_sql_name(col)
                    con.execute(
                        f"ALTER TABLE geschichtsseiten ADD COLUMN {col} {dtype_dict[col]}"
                    )
                new_row.transform(
                    lambda col: col
                    if dtype_dict[col.name] != "BLOB"
                    else col.map(pickle.dumps)
                ).to_sql("geschichtsseiten", con, if_exists="append", dtype=dtype_dict)
        return pd_read_sql(con, "geschichtsseiten", index=index)


def get_global_metadata_df() -> pd.DataFrame:
    try:
        with db_con() as con:
            return pd_read_sql(con, "global")
    except FileNotFoundError as err:
        raise Exception(
            "FileNotFoundError: Likely database is missing. If you haven't, run "
            f"`download_global_metadata(query_dict)`. Original error was:\n{str(err)}"
        )
