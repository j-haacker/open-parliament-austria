"""Pkg doc-str"""

__all__ = []

from collections.abc import Iterable
from aiohttp import ClientSession
from contextlib import contextmanager
from datetime import datetime
import json
import numpy as np
from open_parliament_austria.resources import _sql_keywords
import os
import pandas as pd
from pathlib import Path
import pickle
from pypdf import PdfReader
import requests
import sqlite3
from typing import Any, Literal


def lib_data():
    return Path(os.getenv("OPA_PATH", Path.home() / ".open-parliament-austria"))


def raw_data():
    return lib_data() / "data" / "raw"


def _ensure_allowed_sql_name(name: str):
    if name.upper() in _sql_keywords or not name.replace("_", "").isalnum():
        raise Exception(
            f"Name {name} is currently not allowed (only sepecial characters '_' and "
            "no SQL keywords)."
        )


def _add_missing_db_cols(con: sqlite3.Connection, table_name: str, df: pd.DataFrame):
    db_col_set = set(_get_colnames(con, table_name))
    dtype_dict = {k: _sqlite3_type(df[k].dropna().iloc[0]) for k in df.columns}
    for col in [col for col in df.columns if col not in db_col_set]:
        _ensure_allowed_sql_name(col)
        con.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {dtype_dict[col]}")


def _append_global_metadata(con: sqlite3.Connection, global_metadata_df: pd.DataFrame):
    dtype_dict = {
        k: _sqlite3_type(v)
        for k, v in global_metadata_df.dropna(axis=0).iloc[0].items()
    }
    global_metadata_df.transform(
        lambda col: col if dtype_dict[col.name] != "BLOB" else col.map(pickle.dumps)
    ).to_sql("global", con=con, if_exists="append", dtype=dtype_dict)


def _download_collection_metadata(
    dataset: Literal["antraege"], query_dict: dict | None = None
):
    URL = _prepend_url("/Filter/api/")
    if dataset in ["antraege"]:
        URL += "filter/data/101"
        params = {"js": "eval", "showAll": "true"}  # "export": "true" <- not necessary
        if query_dict is None:
            query_dict = {}
        query_dict.update(VHG=["ANTR"])
    elif dataset in ["sitzungen"]:
        URL += "filter/data/211"
        params = {"js": "eval", "showAll": "true"}  # "export": "true" <- not necessary
    else:
        raise Exception(
            "Implement API parameters for this dataset. See "
            "<https://www.parlament.gv.at/recherchieren/open-data/daten-und-lizenz/index.html> "
            "-> Follow link to the dataset -> View 'Wie funktioniert die API?'"
        )
    res = requests.post(
        URL, params=params, data={} if query_dict is None else json.dumps(query_dict)
    )
    if res.status_code != 200:
        raise Exception(
            f"Failed with {res.status_code}! (url: {res.url}, payload: {query_dict})"
        )
    return res.json()


async def _download_file(session: ClientSession, url: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    async with session.get(url) as res:
        if res.status == 200:
            with open(target, "wb") as f:
                async for chunk in res.content.iter_chunked(4096):
                    f.write(chunk)
        else:
            print(f"! Failed with {res.status}! (url: {res.url})")


def _extract_txt_from_pdf(pdf_file: Path | str):
    reader = PdfReader(pdf_file)
    pages = reader.pages
    return "\n".join([p.extract_text() for p in pages])


def _get_db_connector(file_name: str) -> callable:
    @contextmanager
    def _exp_func():
        raw_data().mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(raw_data() / file_name) as con:
            yield con

    return _exp_func


def _get_rowid_index(
    con: sqlite3.Connection, tbl: str, index_col: tuple[str]
) -> pd.Series:
    return pd.DataFrame(
        con.execute(f"SELECT {' ,'.join(index_col)}, rowid FROM {tbl}").fetchall(),
        columns=[*(index_col), "rowid"],
    ).set_index(list(index_col))["rowid"]


def _get_colnames(con: sqlite3.Connection, tbl: str) -> list[str]:
    return [
        row[0]
        for row in con.execute(
            "SELECT name FROM pragma_table_info(?)", [tbl]
        ).fetchall()
    ]


def _get_colname_by_type(con: sqlite3.Connection, tbl: str, _type: str) -> list[str]:
    return [
        row[0]
        for row in con.execute(
            f"SELECT name FROM pragma_table_info('{tbl}') WHERE type = '{_type.upper()}'"
        ).fetchall()
    ]


def _get_pd_sql_reader(index_col: list[str]):
    def pd_read_sql(
        con: sqlite3.Connection,
        tablename: str,
        columns: Iterable[str] | None = None,
        index: Iterable[tuple[str | int]] | None = None,
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
                lambda col: col
                if col.name not in datetime_cols
                else pd.to_datetime(col)
            )
        )
        return df.transform(
            lambda col: col
            if col.name not in pickled_cols
            else col.map(lambda x: None if x is None else pickle.loads(x))
        )

    return pd_read_sql


def _prepend_url(path: str) -> str:
    return "https://www.parlament.gv.at" + path


def _quote_if_str(x: Any) -> Any:
    return x if not isinstance(x, str) else f"'{x}'"


def _sqlite3_type(x: Any) -> str:
    match x:
        case None:
            return "NULL"
        case int() | np.integer() | bool() | np.bool():
            return "INTEGER"
        case float() | np.floating():
            return "REAL"
        case datetime():
            return "DATETIME"
        case str():
            return "TEXT"
        case set() | list() | tuple() | dict() | np.ndarray():
            return "BLOB"
    # anything can be stored as blob, however for hygiene raise
    raise Exception(f"Couldn't determine sqlite3 type for {x}, class {x.__class__}.")
