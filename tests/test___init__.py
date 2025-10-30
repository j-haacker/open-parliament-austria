import pytest
import open_parliament_austria as opa
from fpdf import FPDF


def test_prepend_url():
    path = "/foo/bar"
    assert opa._prepend_url(path) == "https://www.parlament.gv.at/foo/bar"


def test_extract_txt_from_pdf(tmp_path):
    # Create a simple PDF file
    pdf_path = tmp_path / "test.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Hello PDF", ln=True)
    pdf.output(str(pdf_path))
    text = opa._extract_txt_from_pdf(pdf_path)
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
    result = opa._download_collection_metadata("antraege")
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
        opa._download_collection_metadata("antraege")


def test_download_collection_metadata_invalid_dataset():
    with pytest.raises(Exception):
        opa._download_collection_metadata("invalid")
