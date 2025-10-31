from aiohttp import ClientSession
import asyncio
from datetime import datetime
import numpy as np
from pandas import Timestamp
import pytest
import open_parliament_austria as opa
from open_parliament_austria import (
    _download_collection_metadata,
    _download_file,
    _ensure_allowed_sql_name,
    _extract_txt_from_pdf,
    _prepend_url,
    _quote_if_str,
    _sqlite3_type,
)
from fpdf import FPDF


def test_prepend_url():
    path = "/foo/bar"
    assert _prepend_url(path) == "https://www.parlament.gv.at/foo/bar"


def test__quote_if_str():
    assert _quote_if_str(0) == 0
    assert _quote_if_str("0") == "'0'"


def test_extract_txt_from_pdf(tmp_path):
    # Create a simple PDF file
    pdf_path = tmp_path / "test.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Hello PDF", ln=True)
    pdf.output(str(pdf_path))
    text = _extract_txt_from_pdf(pdf_path)
    assert "Hello PDF" in text


def test_download_collection_metadata_success(monkeypatch):
    class DummyResponse:
        status_code = 200

        def json(self):
            return {"foo": "bar"}

    def dummy_post(url, params, data):
        assert "filter/data/101" in url
        return DummyResponse()

    monkeypatch.setattr(opa.requests, "post", dummy_post)
    result = _download_collection_metadata("antraege")
    assert result == {"foo": "bar"}


def test_download_collection_metadata_fail(monkeypatch):
    class DummyResponse:
        status_code = 404

        def json(self):
            return {}

    def dummy_post(url, params, data):
        return DummyResponse()

    monkeypatch.setattr(opa.requests, "post", dummy_post)
    with pytest.raises(Exception):
        _download_collection_metadata("antraege")


def test_download_collection_metadata_invalid_dataset():
    with pytest.raises(Exception):
        _download_collection_metadata("invalid")


def test_download_file_fail(tmp_path, capsys):
    loop = asyncio.new_event_loop()
    task = loop.create_task(
        _download_file(
            ClientSession(loop=loop), _prepend_url("/does/not/exist"), tmp_path
        )
    )
    loop.run_until_complete(task)
    assert capsys.readouterr().out.startswith("! Failed with ")


def test__ensure_allowed_col_name_fail():
    for s in ["select", "afa!"]:
        with pytest.raises(Exception, match=f"Name {s} is currently not allowed"):
            _ensure_allowed_sql_name(s)


def test__sqlite3_type():
    assert _sqlite3_type(None) == "NULL"
    for x in [1312, np.int16(1), True, np.bool(False)]:
        assert _sqlite3_type(x) == "INTEGER"
    for x in [13.12, np.float16(1)]:
        assert _sqlite3_type(x) == "REAL"
    dt = datetime(year=1312, day=13, month=12, hour=13, minute=12)
    for x in [dt, Timestamp(dt)]:
        assert _sqlite3_type(x) == "DATETIME"
    assert _sqlite3_type("no border\nno nation") == "TEXT"
    for x in [
        {1, "a"},
        [1, "a", [1, "a"]],
        (1, "a", [1, "a"]),
        {1: "a", "b": [1, "a"]},
        np.ndarray([1, 3, 1, 2]),
    ]:
        assert _sqlite3_type(x) == "BLOB"
    with pytest.raises(Exception, match="Couldn't determine sqlite3 type"):
        _sqlite3_type(np.datetime64(dt))
