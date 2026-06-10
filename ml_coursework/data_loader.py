from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from .settings import (
    CHURN_FIGURE_DIR,
    CHURN_DATA_FILE,
    CHURN_DATA_URL,
    CHURN_NOTEBOOK_DIR,
    CHURN_TABLE_DIR,
    CO2_FIGURE_DIR,
    CO2_DATA_FILE,
    CO2_DATA_URL,
    CO2_NOTEBOOK_DIR,
    CO2_TABLE_DIR,
    DATA_DIR,
    FIGURE_DIR,
    NOTEBOOK_DIR,
    REPORT_DIR,
    TABLE_DIR,
)


def ensure_project_directories() -> None:
    for path in (
        DATA_DIR,
        NOTEBOOK_DIR,
        REPORT_DIR,
        FIGURE_DIR,
        TABLE_DIR,
        CO2_NOTEBOOK_DIR,
        CHURN_NOTEBOOK_DIR,
        CO2_FIGURE_DIR,
        CHURN_FIGURE_DIR,
        CO2_TABLE_DIR,
        CHURN_TABLE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> Path:
    ensure_project_directories()
    if destination.exists():
        return destination

    with urlopen(url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)
    return destination


def load_co2_data(force_download: bool = False) -> pd.DataFrame:
    if force_download and CO2_DATA_FILE.exists():
        CO2_DATA_FILE.unlink()
    download_file(CO2_DATA_URL, CO2_DATA_FILE)
    return pd.read_csv(CO2_DATA_FILE)


def load_churn_data(force_download: bool = False) -> pd.DataFrame:
    if force_download and CHURN_DATA_FILE.exists():
        CHURN_DATA_FILE.unlink()
    download_file(CHURN_DATA_URL, CHURN_DATA_FILE)
    return pd.read_csv(CHURN_DATA_FILE)
