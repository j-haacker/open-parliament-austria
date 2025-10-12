"""Pkg doc-str"""

__all__ = []

import json
from pathlib import Path
from pypdf import PdfReader
import requests
from typing import Literal

lib_data = Path.home() / ".open-parliament-austria"


def _extract_txt_from_pdf(pdf_file: Path | str):
    reader = PdfReader(pdf_file)
    pages = reader.pages
    return "\n".join([p.extract_text() for p in pages])


def _prepend_url(path: str) -> str:
    return "https://www.parlament.gv.at" + path


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
        raise Exception(f"Failed with {res.status_code}!")
    return res.json()


def _download_file(url: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    res = requests.get(url, stream=True)
    with open(target, "wb") as f:
        for chunk in res.iter_content(chunk_size=4096):
            if chunk:
                f.write(chunk)
