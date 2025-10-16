"""Pkg doc-str"""

__all__ = []

from datetime import datetime
import json
import numpy as np
from pathlib import Path
from pypdf import PdfReader
import requests
import sqlite3
from typing import Any, Literal

lib_data = Path.home() / ".open-parliament-austria"


def _download_collection_metadata(
    dataset: Literal["antraege"], query_dict: dict | None = None
):
    URL = _prepend_url("/Filter/api/")
    if dataset in ["antraege"]:
        URL += "filter/data/101"
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


def _download_file(url: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    res = requests.get(url, stream=True)
    with open(target, "wb") as f:
        for chunk in res.iter_content(chunk_size=4096):
            if chunk:
                f.write(chunk)


def _extract_txt_from_pdf(pdf_file: Path | str):
    reader = PdfReader(pdf_file)
    pages = reader.pages
    return "\n".join([p.extract_text() for p in pages])


def _get_colnames(con: sqlite3.Connection, tbl: str) -> list[str]:
    return [
        row[0]
        for row in con.execute(
            f"SELECT name FROM pragma_table_info('{tbl}')"
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
        case set() | list() | tuple() | dict():
            return "BLOB"
    # anything can be stored as blob, however for hygiene raise
    raise Exception(f"Couldn't determine sqlite3 type for {x}, class {x.__class__}.")
