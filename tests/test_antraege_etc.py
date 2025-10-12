"""Tests for antraege_etc.py module."""

import json
from pathlib import Path
import pandas as pd
import pytest
import pickle
import sqlite3
from unittest.mock import patch, MagicMock

from open_parliament_austria.antraege_etc import (
    append_global_metadata,
    _build_global_metadataframe_from_json,
    _download_document,
    get_global_metadata_df,
    get_antragstext,
    index_col,
)


@pytest.fixture
def sample_metadata_df():
    """Create a sample metadata DataFrame for testing."""
    data = {
        "GP_CODE": ["27", "27"],
        "ITYP": ["A", "A"],
        "INR": ["1", "2"],
        "datum_eingebracht": ["2023-01-01", "2023-01-02"],
        "HIS_URL": ["/TEST/URL1", "/TEST/URL2"],
        "ist_werteliste": ['["test1", "test2"]', None],
    }
    df = pd.DataFrame(data)
    df["datum_eingebracht"] = pd.to_datetime(df["datum_eingebracht"])
    return df.set_index(index_col)


@pytest.fixture
def sample_json_data():
    """Create sample JSON data for testing."""
    columns = [
        "GP_CODE",
        "ITYP",
        "INR",
        "datum_eingebracht",
        "HIS_URL",
        "ist_werteliste",
        "ZUKZ",
        "RSS_DESC",
        "INRNUM",
        "LZ-Buttons",
        "sysdate???",
        "WENTRY_ID",
        "NR_GP_CODE",
        "Gruppe",
    ]
    return {
        "header": {"label": columns, "ist_werteliste": ["0"] * len(columns)},
        "rows": [
            [
                "27",
                "A",
                "1",
                "2023-01-01",
                "/TEST/URL1",
                '["test1", "test2"]',
                "ZUKZ1",
                "DESC1",
                "1",
                "BTN1",
                "2023-01-01",
                "ID1",
                "27",
                "GRP1",
            ],
            [
                "27",
                "A",
                "2",
                "2023-01-02",
                "/TEST/URL2",
                "null",
                "ZUKZ2",
                "DESC2",
                "2",
                "BTN2",
                "2023-01-02",
                "ID2",
                "27",
                "GRP2",
            ],
        ],
    }


@pytest.fixture
def mock_db_path(tmp_path):
    """Create a temporary database path."""
    db_path = tmp_path / "metadata_api_101.db"
    return db_path


@patch("open_parliament_austria.antraege_etc.raw_data")
def test_append_global_metadata(mock_raw_data, mock_db_path, sample_metadata_df):
    """Test appending metadata to SQLite database."""
    mock_raw_data.__truediv__.return_value = mock_db_path

    append_global_metadata(sample_metadata_df)

    # Verify data was stored correctly
    with sqlite3.connect(mock_db_path) as con:
        df = pd.read_sql("SELECT * FROM global", con=con)
        df = df.map(lambda x: pickle.loads(x) if isinstance(x, bytes) else x)
        df = df.set_index(index_col)

        pd.testing.assert_frame_equal(df, sample_metadata_df)


@patch("open_parliament_austria.antraege_etc.raw_data")
@patch("open_parliament_austria.antraege_etc._download_collection_metadata")
def test_build_global_metadataframe_from_json(
    mock_download, mock_raw_data, tmp_path, sample_json_data
):
    """Test building global metadata DataFrame from JSON."""
    json_path = tmp_path / "global_metadata.json"
    db_path = tmp_path / "metadata_api_101.db"

    def mock_truediv(path):
        path_str = str(path) if isinstance(path, Path) else path
        if path_str == "global_metadata.json":
            return json_path
        return db_path

    mock_raw_data.__truediv__.side_effect = mock_truediv
    mock_download.return_value = sample_json_data

    _build_global_metadataframe_from_json()

    assert json_path.exists()
    with open(json_path) as f:
        saved_data = json.load(f)
    assert saved_data == sample_json_data


@patch("open_parliament_austria.antraege_etc._prepend_url")
@patch("requests.get")
def test_download_document(mock_get, mock_prepend_url, sample_metadata_df, tmp_path):
    """Test downloading a document."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "content": {
            "documents": [
                {"documents": [{"type": "PDF", "link": "/test/document.pdf"}]}
            ]
        }
    }
    mock_get.return_value = mock_response
    mock_prepend_url.side_effect = lambda x: f"https://test.com{x}"

    with (
        patch("open_parliament_austria.antraege_etc.raw_data", tmp_path),
        patch("open_parliament_austria.antraege_etc._download_file"),
    ):
        _download_document(sample_metadata_df.iloc[0])

    assert mock_get.call_count == 1
    mock_get.assert_called_with("https://test.com/TEST/URL1?json=True")
    assert mock_prepend_url.call_count == 2


@patch("open_parliament_austria.antraege_etc.raw_data")
def test_get_global_metadata_df(mock_raw_data, mock_db_path, sample_metadata_df):
    """Test retrieving global metadata DataFrame."""
    mock_raw_data.__truediv__.return_value = mock_db_path

    # First store some data
    append_global_metadata(sample_metadata_df)

    # Now test retrieval
    result = get_global_metadata_df()
    pd.testing.assert_frame_equal(result, sample_metadata_df)


@patch("open_parliament_austria.antraege_etc._download_document")
@patch("open_parliament_austria.antraege_etc._extract_txt_from_pdf")
def test_get_antragstext(mock_extract, mock_download, sample_metadata_df, tmp_path):
    """Test retrieving Antragstext."""
    test_text = "Sample text content"
    mock_extract.return_value = test_text

    # Create test path structure
    doc_path = tmp_path / "27" / "A" / "1"
    doc_path.mkdir(parents=True)
    pdf_path = doc_path / "document.pdf"
    pdf_path.touch()

    with patch("open_parliament_austria.antraege_etc.raw_data", tmp_path):
        # First call should create txt file
        result = get_antragstext(sample_metadata_df.iloc[0])
        assert result == test_text

        # Second call should read existing txt file
        result = get_antragstext(sample_metadata_df.iloc[0])
        assert result == test_text

    mock_extract.assert_called_once_with(pdf_path)
    mock_download.assert_not_called()  # Because file already existed


@patch("open_parliament_austria.antraege_etc._prepend_url")
@patch("requests.get")
def test_download_document_no_docs(mock_get, mock_prepend_url, sample_metadata_df):
    """Test download_document when no documents are available."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"content": {"documents": []}}
    mock_get.return_value = mock_response
    mock_prepend_url.side_effect = lambda x: f"https://test.com{x}"

    with pytest.raises(Exception, match="No docs for"):
        _download_document(sample_metadata_df.iloc[0])


@patch("open_parliament_austria.antraege_etc._prepend_url")
@patch("requests.get")
def test_download_document_multiple_docs(
    mock_get, mock_prepend_url, sample_metadata_df
):
    """Test download_document when multiple documents are available."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "content": {"documents": [{"documents": []}, {"documents": []}]}
    }
    mock_get.return_value = mock_response
    mock_prepend_url.side_effect = lambda x: f"https://test.com{x}"

    with pytest.raises(Exception, match="Not implmented"):
        _download_document(sample_metadata_df.iloc[0])


@patch("open_parliament_austria.antraege_etc._prepend_url")
@patch("requests.get")
def test_download_document_non_pdf(mock_get, mock_prepend_url, sample_metadata_df):
    """Test download_document when document is not PDF."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "content": {
            "documents": [
                {"documents": [{"type": "DOCX", "link": "/test/document.docx"}]}
            ]
        }
    }
    mock_get.return_value = mock_response
    mock_prepend_url.side_effect = lambda x: f"https://test.com{x}"

    with pytest.raises(Exception, match="Not implemented for type"):
        _download_document(sample_metadata_df.iloc[0])
