"""Pkg doc-str"""

__all__ = []

from collections.abc import Iterable
from aiohttp import ClientSession
from contextlib import contextmanager
from datetime import datetime
import json
import numpy as np
from open_parliament_austria.resources import _column_name_dict, _sql_keywords
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


def _create_global_db_tbl(
    con: sqlite3.Connection, index_col: list[str], index_sqltypes: list[str]
):
    [_ensure_allowed_sql_name(c) for c in index_col]
    for t in index_sqltypes:
        if t not in ["INTEGER", "TEXT", "NUMERIC"]:
            raise Exception(f"Type {t} is not supported as index column.")
    sql = (
        "CREATE TABLE IF NOT EXISTS global("
        + ", ".join([f"{c} {t}" for c, t in zip(index_col, index_sqltypes)])
        + "); CREATE UNIQUE INDEX IF NOT EXISTS ix_global_"
        + "_".join([f"{c}" for c in index_col])
        + " ON global("
        + ", ".join(index_col)
        + ");"
    )
    # print(sql.format(*index_col))
    con.executescript(sql)


def _deflate_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == np.dtype("O"):
            # print(f"{col} o-type")
            if not df[col].dropna().empty and df[col].dropna().str.match(" *\{").all():
                # print(col, "matched")
                # print(df[col].dropna())
                df = df.drop(columns=col).join(
                    pd.DataFrame.from_records(df[col]), rsuffix="_attr"
                )
        # else:
        #     print(col, df[col].dtype)
    return df


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
        params = {"js": "eval", "showAll": "true"}
    elif dataset in ["personen"]:
        URL += "filter/data/409"
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


def _get_coll_downloader(
    db_con: callable,
    dataset: Literal["antraege", "sitzungen", "personen"],
    index_col: list[str],
    index_sqltypes: list[str],
):
    def _exp_func(query_dict: dict | None = None):
        _json = _download_collection_metadata(dataset=dataset, query_dict=query_dict)
        # print(f"{_json["count"]} rows")
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
                if col not in _column_name_dict
            ],
            inplace=True,
        )

        ## polish table
        global_metadata_df = _deflate_columns(
            global_metadata_df
        )  # before null conversion
        global_metadata_df = global_metadata_df.apply(
            lambda col: col
            if not col.dtype == np.dtype("O")
            else col.str.replace("null", "None")
        )
        global_metadata_df.rename(columns=_column_name_dict, inplace=True)
        date_col = [col for col in global_metadata_df.columns if "datum" in col.lower()]
        global_metadata_df[date_col] = global_metadata_df[date_col].apply(
            pd.to_datetime
        )
        global_metadata_df = global_metadata_df.apply(
            lambda col: col
            if not col.dtype == np.dtype("O") or not all(col.str.isdecimal())
            else col.astype(int)
        )
        if "ist_werteliste" in header:
            list_col = [
                _column_name_dict[idx]
                for idx, val in header.ist_werteliste.items()
                if val
            ]
            global_metadata_df[list_col] = global_metadata_df[list_col].map(
                lambda x: x
                if x is None
                else np.array(
                    [y.strip() for y in x.split(",")] if "[" not in x else json.loads(x)
                )
            )
        print(global_metadata_df.columns)
        global_metadata_df.set_index(index_col, inplace=True)
        global_metadata_df.sort_index(inplace=True)
        global_metadata_df.dropna(axis=1, how="all", inplace=True)
        # print(global_metadata_df.to_string())

        ## write to db
        # with db_con() as con:
        #     print(_get_colnames(con, "global"))
        # _create_global_db_tbl()
        with db_con() as con:
            _create_global_db_tbl(con, index_col, index_sqltypes)
            _add_missing_db_cols(con, "global", global_metadata_df)
            _append_global_metadata(con, global_metadata_df)

    return _exp_func


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
