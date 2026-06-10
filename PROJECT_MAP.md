# Project Map

This document provides a concise guide to the repository structure, execution flow, and generated artifacts.

## Main Execution Commands

Run the CO2 regression pipeline:

```powershell
python -m ml_coursework.co2
```

Run the customer churn classification pipeline:

```powershell
python -m ml_coursework.churn
```

Run a pipeline with the interactive figure viewer:

```powershell
python -m ml_coursework.co2 --show
python -m ml_coursework.churn --show
```

## Source Code

| File | Role |
| --- | --- |
| `ml_coursework/co2.py` | End-to-end regression workflow for vehicle CO2 emissions. |
| `ml_coursework/churn.py` | End-to-end binary classification workflow for telecom churn prediction. |
| `ml_coursework/data_loader.py` | Centralized dataset loading and validation utilities. |
| `ml_coursework/settings.py` | Shared project paths, random seed, train/test ratio, and cross-validation settings. |
| `ml_coursework/visual_style.py` | Common chart theme and plotting utilities. |
| `ml_coursework/figure_viewer.py` | Viewer for browsing generated figures in a single window. |

## Data

| Path | Description |
| --- | --- |
| `data/raw/FuelConsumptionCo2.csv` | Vehicle fuel consumption and CO2 emissions dataset. |
| `data/raw/Telco-Customer-Churn.csv` | Telecom customer churn dataset. |

## Notebooks

| Notebook | Description |
| --- | --- |
| `notebooks/co2/co2.ipynb` | Reproducible notebook for the CO2 regression project. |
| `notebooks/churn/churn.ipynb` | Reproducible notebook for the churn classification project. |

The notebooks mirror the Python pipelines and present the analysis in a report-friendly format.

## Reports

| Path | Content |
| --- | --- |
| `reports/figures/co2` | CO2 EDA charts, model comparison, residual analysis, and feature importance. |
| `reports/figures/churn` | Churn EDA charts, model comparison, threshold analysis, ROC/PR curves, and confusion matrix. |
| `reports/tables/co2` | Regression metrics, cross-validation results, tuning results, and error analysis. |
| `reports/tables/churn` | Classification metrics, cross-validation results, threshold optimization, and error analysis. |
| `reports/documents` | Technical documentation for the project. |

## Submission Folders

| Path | Content |
| --- | --- |
| `submissions/co2` | Standalone materials for the CO2 regression task. |
| `submissions/churn` | Standalone materials for the churn classification task. |

Each submission folder contains the relevant notebook, dataset, generated figures, evaluation tables, and dependency file.

## Execution Flow

```text
Raw data
→ Data loading and validation
→ Exploratory data analysis
→ Preprocessing pipeline
→ Train/test split
→ Cross-validation model comparison
→ Hyperparameter tuning
→ Final test evaluation
→ Figure and table export
```

## Design Notes

- The two projects are independent but share the same coding style and project structure.
- Configuration values are centralized in `settings.py` to keep execution reproducible.
- The code separates reusable utilities from project-specific modeling logic.
- Generated reports are stored outside the source code package to keep implementation and outputs clearly separated.
