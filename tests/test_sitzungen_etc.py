"""Tests for antraege_etc.py module."""

# import json
# from pathlib import Path
# import pandas as pd
import pytest

# import pickle
# import sqlite3
# from unittest.mock import patch, MagicMock
from open_parliament_austria.sitzungen_etc import (
    download_global_metadata,
)


@pytest.fixture(scope="module")
def init_db():
    download_global_metadata(  # "GP_CODE": ["XXVII"], "DOKTYP": ["NRSITZ"],
        {"DATUM": ["2025-07-01", "2025-07-10"]}
    )


def test_tmp(init_db):
    assert True
