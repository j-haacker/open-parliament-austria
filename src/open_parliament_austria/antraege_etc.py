"""Module for requests to API endpoint 101

This includes:

- Antraege
- ...

The collected metadata focus on "Antraege" and associated processes.
Choices may need to change in future to support more datasets.
"""

__all__ = []

from ast import literal_eval
from bs4 import BeautifulSoup
import numpy as np
from open_parliament_austria import (
    _download_collection_metadata,
    _download_file,
    _extract_txt_from_pdf,
    _get_colnames,
    _get_colname_by_type,
    lib_data,
    _prepend_url,
    _sqlite3_type,
)
import pandas as pd
from pathlib import Path
import pickle
import requests
import sqlite3
from typing import Any, Literal

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
    print(_dict)
    if "documents" in _dict["content"]:
        docs = _dict["content"]["documents"]
    print(docs)
    if "docs" not in locals() or len(docs) == 0:
        raise Exception(f"No docs for {df_row.name}.")
    elif len(docs) > 2:
        # See what docs are included and implement handling
        raise Exception(f"Not implmented. Docs:\n{repr(docs)}")
    elif len(docs) > 1:
        # prefer digital version
        if any(["(elektr. übermittelte Version)" in doc["title"] for doc in docs]):
            docs = next(
                doc["documents"]
                for doc in docs
                if "(elektr. übermittelte Version)" in doc["title"]
            )
        # otherwise select original
        elif any(["(gescanntes Original)" in doc["title"] for doc in docs]):
            docs = next(
                doc["documents"]
                for doc in docs
                if "(gescanntes Original)" in doc["title"]
            )
        else:
            docs = docs[0]["documents"]
    if any(["HTML" in d["type"] for d in docs]):
        link = next(d["link"] for d in docs if "HTML" in d["type"])
    else:
        doc = docs[0]
        if docs["type"] != "PDF":
            raise Exception(f"Not implemented for type {doc['type']}.")
        link = doc["link"]
    path = Path(raw_data, *list(map(str, df_row.name)))
    _download_file(_prepend_url(link), path / link.split("/")[-1])


def pd_read_sql(
    con: sqlite3.Connection,
    tablename: str,
    columns: list[str] | None = None,
    index: tuple[str, str, int] | None = None,
) -> pd.DataFrame:
    datetime_cols = _get_colname_by_type(con, tablename, "datetime")
    pickled_cols = _get_colname_by_type(con, tablename, "blob")
    kwargs = dict(
        con=con,
        index_col=index_col,
        parse_dates=datetime_cols,
    )
    if index is None and columns is None:
        df = pd.read_sql(tablename, **kwargs)
    elif index is None:
        df = pd.read_sql(
            f"SELECT {'*' if columns is None else ', '.join(columns)} FROM {tablename}",
            **kwargs,
        )
    elif len(index) == 3 and isinstance(index[0], str):  # then its a single index
        df = pd.read_sql(
            f"SELECT {'*' if columns is None else ', '.join(columns)} "
            f"FROM {tablename} WHERE "
            + " AND ".join([f"{col} = '{val}'" for col, val in zip(index_col, index)]),
            **kwargs,
        )
    else:
        raise Exception("Not implemented to extract multiple rows.")
    return df.transform(
        lambda col: col
        if col.name not in pickled_cols
        else col.map(lambda x: None if x is None else pickle.loads(x))
    )


def _quote_if_str(x: Any) -> str:
    return f"{x}" if not isinstance(x, str) else f"'{x}'"


def get_geschichtsseite(df_row: pd.Series) -> pd.DataFrame:
    if not (raw_data / "metadata_api_101.db").is_file():
        raise Exception(
            "Error: Database not found. Initialize database using "
            "`get_global_metadata_df(dataset, query)`."
        )
    with sqlite3.connect(raw_data / "metadata_api_101.db") as con:
        tbl_exists = (
            con.execute(
                "SELECT 1 FROM sqlite_master WHERE name='geschichtsseiten'"
            ).fetchone()
            is not None
        )
        if (
            not tbl_exists
            or con.execute(
                "SELECT 1 FROM geschichtsseiten "
                "WHERE "
                + " AND ".join(
                    [
                        f"{col} = {_quote_if_str(val)}"
                        for col, val in zip(index_col, df_row.name)
                    ]
                )
            ).fetchone()
            is None
        ):
            url = _prepend_url(df_row.HIS_URL) + "?json=True"
            _dict = requests.get(url).json()["content"]
            [
                _dict.pop(k)
                for k in [
                    *[
                        ix.lower()
                        for ix in df_row.index.to_list() + index_col
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
                index=pd.MultiIndex.from_tuples((df_row.name,), names=index_col),
            )
            new_row[["update", "einlangen"]] = new_row[
                ["update", "einlangen"]
            ].transform(pd.to_datetime)
            dtype_dict = {k: _sqlite3_type(v) for k, v in new_row.iloc[0].items()}
            if tbl_exists:
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
                    if not col.replace("_", "").isalnum():
                        raise Exception(
                            f"Column name {col} is currently not allowed (only sepecial "
                            "characters '_')."
                        )
                    con.execute(
                        f"ALTER TABLE geschichtsseiten ADD COLUMN {col} {dtype_dict[col]}"
                    )
            new_row.transform(
                lambda col: col
                if dtype_dict[col.name] != "BLOB"
                else col.map(pickle.dumps)
            ).to_sql("geschichtsseiten", con, if_exists="append", dtype=dtype_dict)
        return pd_read_sql(con, "geschichtsseiten", index=df_row.name)


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
        for file in path.glob("*.html"):
            with open(file, "r") as f_in, open(file.with_suffix(".txt"), "w") as f_out:
                f_out.write(BeautifulSoup(f_in.read(), "html.parser").body.text)
    with open(next(path.glob("*.txt")), "r") as f:
        return f.read()
