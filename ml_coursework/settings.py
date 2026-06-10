"""Shared project settings."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
NOTEBOOK_DIR = ROOT_DIR / "notebooks"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
CO2_NOTEBOOK_DIR = NOTEBOOK_DIR / "co2"
CHURN_NOTEBOOK_DIR = NOTEBOOK_DIR / "churn"
CO2_FIGURE_DIR = FIGURE_DIR / "co2"
CHURN_FIGURE_DIR = FIGURE_DIR / "churn"
CO2_TABLE_DIR = TABLE_DIR / "co2"
CHURN_TABLE_DIR = TABLE_DIR / "churn"

CO2_DATA_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/"
    "FuelConsumptionCo2.csv"
)
CHURN_DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

CO2_DATA_FILE = DATA_DIR / "FuelConsumptionCo2.csv"
CHURN_DATA_FILE = DATA_DIR / "Telco-Customer-Churn.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TOP_FEATURES_TO_SHOW = 15
