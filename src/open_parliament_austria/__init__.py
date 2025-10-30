"""Pkg doc-str"""

__all__ = []

from aiohttp import ClientSession
from datetime import datetime
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
from pypdf import PdfReader
import requests
import sqlite3
from typing import Any, Literal


def lib_data():
    return Path(os.getenv("OPA_PATH", Path.home() / ".open-parliament-austria"))


def raw_data():
    return lib_data() / "data" / "raw"


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
                    if chunk:
                        f.write(chunk)
        else:
            print(f"! Failed with {res.status}! (url: {res.url})")


def _extract_txt_from_pdf(pdf_file: Path | str):
    reader = PdfReader(pdf_file)
    pages = reader.pages
    return "\n".join([p.extract_text() for p in pages])


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


def _prepend_url(path: str) -> str:
    return "https://www.parlament.gv.at" + path


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
