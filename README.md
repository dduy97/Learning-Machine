# Machine Learning Projects: Vehicle CO2 Regression and Customer Churn Classification

This repository contains two end-to-end machine learning projects developed with a consistent workflow: data exploration, preprocessing, model comparison, hyperparameter tuning, final evaluation, and report artifact generation.

The project is organized as a reproducible analysis package. Each task can be executed independently from the command line and produces its own figures, evaluation tables, notebooks, and submission-ready outputs.

## Project Overview

| Project | Task Type | Objective | Final Model |
| --- | --- | --- | --- |
| Vehicle CO2 Emissions | Regression | Predict vehicle CO2 emissions from technical and fuel-consumption attributes. | Linear Regression |
| Customer Churn | Binary Classification | Predict whether a telecom customer is likely to leave the service. | Logistic Regression |

## Key Results

### Vehicle CO2 Emissions

The CO2 regression project achieved strong predictive performance because vehicle emissions are highly related to fuel consumption and engine characteristics.

| Metric | Value |
| --- | ---: |
| MAE | 3.56 |
| RMSE | 5.45 |
| R2 | 0.9921 |

### Customer Churn

The churn classification project focuses on identifying customers at risk of leaving. The final decision threshold was selected using validation performance to improve the balance between recall and F1-score.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7495 |
| Precision | 0.5182 |
| Recall | 0.7661 |
| F1-score | 0.6182 |
| ROC-AUC | 0.8398 |

## Methodology

Both projects follow the same machine learning pipeline:

1. Load raw datasets from `data/raw`.
2. Inspect data quality, distributions, outliers, and feature relationships.
3. Build exploratory visualizations for numerical and categorical variables.
4. Split the dataset into training and testing sets using an 80/20 ratio.
5. Apply preprocessing with scikit-learn pipelines.
6. Compare multiple candidate models using cross-validation.
7. Tune selected models with randomized hyperparameter search.
8. Evaluate the best model on the hold-out test set.
9. Export charts, metrics, model comparison tables, and analysis artifacts.

## Repository Structure

```text
A49932_hocmay/
├── data/raw/                 # Original datasets
├── ml_coursework/            # Main Python package
│   ├── co2.py                # CO2 regression pipeline
│   ├── churn.py              # Churn classification pipeline
│   ├── data_loader.py        # Dataset loading utilities
│   ├── figure_viewer.py      # Interactive figure viewer
│   ├── settings.py           # Shared paths and configuration
│   └── visual_style.py       # Shared chart styling
├── notebooks/                # Executable analysis notebooks
├── reports/
│   ├── documents/            # Technical report documents
│   ├── figures/              # Generated visualizations
│   └── tables/               # Generated evaluation tables
├── submissions/              # Separated submission folders
├── PROJECT_MAP.md            # Detailed project map
└── requirements.txt          # Python dependencies
```

## Installation

Create and activate a virtual environment, then install the required libraries.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Projects

Run the CO2 regression project:

```powershell
python -m ml_coursework.co2
```

Run the churn classification project:

```powershell
python -m ml_coursework.churn
```

Run a project and open the interactive chart viewer:

```powershell
python -m ml_coursework.co2 --show
python -m ml_coursework.churn --show
```

## Output Artifacts

After execution, the project writes results into the `reports` directory:

| Directory | Content |
| --- | --- |
| `reports/figures/co2` | EDA charts, model comparison charts, residual analysis, and feature importance for CO2 regression. |
| `reports/figures/churn` | Class distribution, churn-rate analysis, ROC/PR curves, confusion matrix, and feature importance for churn classification. |
| `reports/tables/co2` | Regression metrics, cross-validation results, tuning comparison, and error analysis. |
| `reports/tables/churn` | Classification metrics, cross-validation results, threshold optimization, and error analysis. |
| `reports/documents` | Technical project documentation. |

## Technical Highlights

- Reproducible train/test split through a fixed random seed.
- Separate pipelines for regression and classification tasks.
- Numerical and categorical preprocessing handled inside scikit-learn pipelines.
- Cross-validation used before final test evaluation to reduce overfitting risk.
- Hyperparameter tuning included for model selection and performance improvement.
- Clear separation between source code, raw data, notebooks, reports, and submission outputs.

## Dependencies

The project uses the following core libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## License and Data

The repository is prepared for academic coursework and demonstration purposes. The datasets are included locally under `data/raw` to keep the project reproducible.
