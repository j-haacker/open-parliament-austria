"""Tests for antraege_etc.py module."""

import pytest
from open_parliament_austria import _get_db_connector
from open_parliament_austria.antraege import (
    download_global_metadata,
    get_antragstext,
    get_geschichtsseiten,
    get_global_metadata_df,
)


@pytest.fixture(scope="module")
def init_db():
    download_global_metadata(
        {"GP_CODE": ["XXVII"], "DOKTYP": ["A"], "INRNUM": ["1", "10"]}
    )


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


def test_get_geschichtsseiten_fail(monkeypatch):
    import open_parliament_austria.antraege as mod

    monkeypatch.setattr(mod, "db_con", _get_db_connector("xyz.db"))
    with pytest.raises(Exception, match="Error: Database not found."):
        get_geschichtsseiten([("XXVII", "A", "5")])
