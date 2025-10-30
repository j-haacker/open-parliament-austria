import os
import pytest


@pytest.fixture(autouse=True, scope="session")
def redirect_data(tmp_path_factory):
    os.environ["OPA_PATH"] = str(tmp_path_factory.getbasetemp())
