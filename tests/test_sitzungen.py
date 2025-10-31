"""Tests for antraege_etc.py module."""

# import json
# from pathlib import Path
# import pandas as pd
import pytest

# import pickle
# import sqlite3
# from unittest.mock import patch, MagicMock
from open_parliament_austria.sitzungen import (
    download_global_metadata,
    get_global_metadata_df,
    get_geschichtsseiten,
)


@pytest.fixture(scope="module")
def init_db():
    download_global_metadata(  # "GP_CODE": ["XXVII"], "DOKTYP": ["NRSITZ"],
        {"DATUM": ["2022-07-01", "2022-07-10"]}
    )


# def test_tmp(init_db):
#     print(get_global_metadata_df().to_string())
#     # raise
#     assert True


def test_get_geschichtsseiten(init_db):
    idx = get_global_metadata_df().index[[3]]
    print(get_geschichtsseiten(idx).to_string())
    raise
    assert True
