"""Tests for antraege_etc.py module."""

# import json
# from pathlib import Path
# import pandas as pd
import pytest

# import pickle
# import sqlite3
# from unittest.mock import patch, MagicMock
from open_parliament_austria.personen import (
    download_global_metadata,
)


@pytest.fixture(scope="module")
def init_db():
    download_global_metadata(
        {
            # "Gremium": ["NR"],
            "ATTR_JSON.mandate_detail.wahlpartei_code": ["FPÖ"],
            "ATTR_JSON.mandate_detail.gp_code": ["XXVIII"],
        }
    )


def test_tmp(init_db):
    assert True
