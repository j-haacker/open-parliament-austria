"""Tests for antraege_etc.py module."""

# import json
# from pathlib import Path
# import pandas as pd
import pytest
# import pickle
# import sqlite3
# from unittest.mock import patch, MagicMock

from open_parliament_austria.antraege_etc import (
    # append_global_metadata,
    db_path,
    download_global_metadata,
    # _download_document,
    get_geschichtsseiten,
    get_global_metadata_df,
    get_antragstext,
    # get_db_connection,
    # index_col,
    # pd_read_sql,
    _quote_if_str,
    raw_data,
)


def test_db_path():
    assert db_path().parent == raw_data()


def test__quote_if_str():
    assert _quote_if_str(0) == 0
    assert _quote_if_str("0") == "'0'"


@pytest.fixture(scope="module")
def init_db():
    download_global_metadata(
        {"GP_CODE": ["XXVII"], "DOKTYP": ["A"], "INRNUM": ["1", "10"]}
    )


# def test_download_global_metadata(init_db): implicit


# def test_pd_read_sql(init_db): implicit
#     with get_db_connection() as con:
#         pd_read_sql(con, "global")


# def test_get_global_metadata_df(init_db): implicit
#     tmp = get_global_metadata_df()
#     assert isinstance(tmp, pd.DataFrame)
#     assert not tmp.empty
#     assert tmp.index.names == index_col


# def test_get_geschichtsseiten implicit


def test_get_antragstext(init_db):
    idx = get_global_metadata_df().index[[0]]
    file_name = (
        get_geschichtsseiten(idx)
        .iloc[0]
        .documents[0]["documents"][0]["link"]
        .split("/")[-1]
    )
    get_antragstext(idx[0], file_name)  # test both branches: download and preexisting
    tmp = get_antragstext(idx[0], file_name)
    assert isinstance(tmp, str)
    assert len(tmp) > 50
