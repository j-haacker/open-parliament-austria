"""Tests for reden.py module."""

# import json
# from pathlib import Path
# import pandas as pd
import pytest

# import pickle
# import sqlite3
# from unittest.mock import patch, MagicMock
from open_parliament_austria.reden import (
    download_global_metadata,
)


@pytest.fixture(scope="module")
def init_db():
    download_global_metadata(
        {"GP_CODE": ["XXVIII"], "GREMIUM": ["N"], "PAD_INTERN": [35520]}
    )


def test_tmp(init_db):
    assert True
