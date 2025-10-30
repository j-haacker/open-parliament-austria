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
from ast import literal_eval
from bs4 import BeautifulSoup
from collections.abc import Iterable
from contextlib import contextmanager
import numpy as np
from open_parliament_austria import (
    _download_collection_metadata,
    _download_file,
    _extract_txt_from_pdf,
    _get_rowid_index,
    _get_colnames,
    _get_colname_by_type,
    raw_data,
    _prepend_url,
    _sql_keywords,
    _sqlite3_type,
)
import pandas as pd
from pathlib import Path
import pickle
import requests
import sqlite3
from typing import Any, Literal

sqlite3.register_adapter(np.int_, lambda i: int(i))

index_col = ["GP_CODE", "ITYP", "INR"]


def _add_missing_db_cols(
    table_name: str, df: pd.DataFrame, con: sqlite3.Connection | None = None
):
    def _inner(con):
        db_col_set = set(_get_colnames(con, table_name))
        dtype_dict = {k: _sqlite3_type(df[k].dropna().iloc[0]) for k in df.columns}
        for col in [col for col in df.columns if col not in db_col_set]:
            _ensure_allowed_col_name(col)
            con.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {dtype_dict[col]}")

    if con:
        return _inner(con)
    with get_db_connection() as con:
        return _inner(con)


def append_global_metadata(global_metadata_df: pd.DataFrame):
    dtype_dict = {
        k: _sqlite3_type(v)
        for k, v in global_metadata_df.dropna(axis=0).iloc[0].items()
    }
    with get_db_connection() as con:
        global_metadata_df.transform(
            lambda col: col if dtype_dict[col.name] != "BLOB" else col.map(pickle.dumps)
        ).to_sql("global", con=con, if_exists="append", dtype=dtype_dict)


def download_global_metadata(query_dict: dict | None = None):
    #  API results seem to undergo overhaul. column labels change
    _json = _download_collection_metadata(dataset="antraege", query_dict=query_dict)
    header = (
        pd.DataFrame.from_dict(_json["header"])
        .apply(lambda x: x == "1" if x.name.startswith("ist_") else x)
        .iloc[: len(_json["rows"][0])]
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
                "Zustimmung aktiv",  # always ZZZZ ?
                "Zusatz",
                "RSS",
                "INRNUM",  # same as INR
                "LZ-Buttons",  # html button
                "sysdate???",  # download timestamp
                "WENTRY_ID",
                "NRGP",  # same as GP_CODE
                "Gruppe",
            ]
        )
        .rename(
            columns={  # were renamed
                "GP": "GP_CODE",
                "Datum (Sort)": "DATUMSORT",
                "Phasen": "PHASEN_BIS",
                "Klub/Fraktion": "Frak",
                "Geschichtsseite_Url": "HIS_URL",
            }
        )
        .set_index(index_col)
        .sort_index()
    )
    _create_global_db_tbl()
    _add_missing_db_cols("global", global_metadata_df)

    append_global_metadata(global_metadata_df)


def _create_global_db_tbl():
    with get_db_connection() as con:
        sql = (
            "CREATE TABLE IF NOT EXISTS global({0} TEXT, {1} TEXT, {2} INTEGER); "
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_global_GP_CODE_ITYP_INR ON global({0}, {1}, {2});"
        )
        con.executescript(sql.format(*index_col))


def _create_child_db_tbl(
    table_name: str, columns: list[str] = [], _types: list[str] = []
):
    for name in [table_name, *columns, *_types]:
        _ensure_allowed_col_name(name)
    sql = (
        "CREATE TABLE IF NOT EXISTS {3}("
        "{0} TEXT, {1} TEXT, {2} INTEGER, "
        + ", ".join([f"{c} {t}" for c, t in zip(columns, _types)])
        + "FOREIGN KEY ({0}, {1}, {2}) REFERENCES global({0}, {1}, {2})"
        ")"
    ).format(*index_col, table_name)
    with get_db_connection() as con:
        con.execute(sql)


def _create_raw_text_db_tbl():
    with get_db_connection() as con:
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


def _ensure_allowed_col_name(col: str):
    if col.upper() in _sql_keywords or not col.replace("_", "").isalnum():
        raise Exception(
            f"Column name {col} is currently not allowed (only sepecial "
            "characters '_' and no SQL keywords)."
        )


@contextmanager
def get_db_connection():
    with sqlite3.connect(raw_data / "metadata_api_101.db") as con:
        yield con


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
    with get_db_connection() as con:
        return _inner(con)


def pd_read_sql(
    con: sqlite3.Connection,
    tablename: str,
    columns: Iterable[str] | None = None,
    index: Iterable[tuple[str, str, int]] | None = None,
) -> pd.DataFrame:
    datetime_cols = _get_colname_by_type(con, tablename, "datetime")
    pickled_cols = _get_colname_by_type(con, tablename, "blob")
    if columns is None:
        columns = _get_colnames(con, tablename)
    else:
        columns = index_col + columns
    query = f"SELECT {', '.join(columns)} FROM {tablename}"
    if index is not None:
        rowid = _get_rowid_index(con, tablename, tuple(index_col))
        query += f" WHERE rowid IN ({', '.join(map(str, rowid.loc[index].values))})"
    df = (
        pd.DataFrame(con.execute(query).fetchall(), columns=columns)
        .set_index(index_col)
        .transform(
            lambda col: col if col.name not in datetime_cols else pd.to_datetime(col)
        )
    )
    return df.transform(
        lambda col: col
        if col.name not in pickled_cols
        else col.map(lambda x: None if x is None else pickle.loads(x))
    )


def _quote_if_str(x: Any) -> Any:
    return x if not isinstance(x, str) else f"'{x}'"


def get_antragstext(idx: tuple[str, str, int], file_name: str) -> str:
    query = "SELECT raw_text FROM raw_text WHERE" + 4 * " {} = ? AND"

    def _fetch():
        return con.execute(
            query[:-4].format(*index_col, "file_name"), [*idx, file_name]
        ).fetchone()

    with get_db_connection() as con:
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
    if not (raw_data / "metadata_api_101.db").is_file():
        raise Exception(
            "Error: Database not found. Initialize database using "
            "`get_global_metadata_df(dataset, query)`."
        )
    with get_db_connection() as con:
        if (
            con.execute(
                "SELECT 1 FROM sqlite_master WHERE name='geschichtsseiten'"
            ).fetchone()
            is None
        ):
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
                    _ensure_allowed_col_name(col)
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
        with get_db_connection() as con:
            return pd_read_sql(con, "global")
    except FileNotFoundError as err:
        raise Exception(
            "FileNotFoundError: Likely database is missing. If you haven't, run "
            f"`download_global_metadata(query_dict)`. Original error was:\n{str(err)}"
        )
