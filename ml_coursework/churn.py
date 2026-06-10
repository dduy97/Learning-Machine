"""Customer churn classification project.

Run with:
    python -m ml_coursework.churn
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .settings import (
    CHURN_FIGURE_DIR,
    CHURN_TABLE_DIR,
    CV_FOLDS,
    RANDOM_STATE,
    TEST_SIZE,
    TOP_FEATURES_TO_SHOW,
)
from .data_loader import ensure_project_directories, load_churn_data
from .visual_style import (
    BLUE,
    GREEN,
    QUALITATIVE_PALETTE,
    RED,
    SEQUENTIAL_PALETTE,
    TEAL,
    WARM_PALETTE,
    add_bar_labels,
    apply_project_style,
    polish_axes,
    save_chart,
)


TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["customerID"]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


@dataclass
class ClassificationArtifacts:
    cv_results: pd.DataFrame
    tuned_metrics: pd.DataFrame
    optimization_comparison: pd.DataFrame
    feature_importance: pd.DataFrame
    classification_report_df: pd.DataFrame
    error_analysis: pd.DataFrame
    best_model_name: str
    best_params: dict
    best_threshold: float
    figures: dict[str, str]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, object]:
    return {
        "Dummy Classifier": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def prepare_dataset(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = dataframe.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    df = df.drop(columns=DROP_COLUMNS)
    df = df.drop_duplicates().reset_index(drop=True)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    return X, y


def evaluate_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    results = []
    preprocessor = build_preprocessor()
    precision_scorer = make_scorer(precision_score, zero_division=0)
    recall_scorer = make_scorer(recall_score, zero_division=0)
    f1_scorer = make_scorer(f1_score, zero_division=0)

    for model_name, estimator in build_models().items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", estimator)]
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=CV_FOLDS,
            scoring={
                "accuracy": "accuracy",
                "precision": precision_scorer,
                "recall": recall_scorer,
                "f1": f1_scorer,
                "roc_auc": "roc_auc",
            },
            n_jobs=-1,
        )
        results.append(
            {
                "model": model_name,
                "cv_accuracy_mean": scores["test_accuracy"].mean(),
                "cv_precision_mean": scores["test_precision"].mean(),
                "cv_recall_mean": scores["test_recall"].mean(),
                "cv_f1_mean": scores["test_f1"].mean(),
                "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["cv_f1_mean", "cv_roc_auc_mean"], ascending=False
    )
    return results_df.reset_index(drop=True)


def tune_best_model(
    model_name: str, X_train: pd.DataFrame, y_train: pd.Series
) -> RandomizedSearchCV:
    preprocessor = build_preprocessor()
    search_spaces = {
        "Logistic Regression": {
            "model__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            "model__solver": ["lbfgs"],
        },
        "Random Forest": {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 8, 12, 18],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.7, None],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0],
        },
    }

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", build_models()[model_name])]
    )
    search_space = search_spaces[model_name]
    max_combinations = math.prod(len(values) for values in search_space.values())
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_space,
        n_iter=min(12, max_combinations),
        cv=CV_FOLDS,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def extract_feature_importance(search: RandomizedSearchCV) -> pd.DataFrame:
    fitted_pipeline = search.best_estimator_
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        raw_values = model.feature_importances_
    else:
        raw_values = np.abs(model.coef_[0])

    importance = pd.DataFrame(
        {"feature": feature_names, "importance": raw_values}
    ).sort_values(by="importance", ascending=False)
    return importance.head(TOP_FEATURES_TO_SHOW).reset_index(drop=True)


def find_best_threshold(
    fitted_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> float:
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    fitted_pipeline.fit(X_fit, y_fit)
    probabilities = fitted_pipeline.predict_proba(X_valid)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_valid, probabilities)

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best_index = int(np.argmax(f1_scores))
    return float(thresholds[best_index])


def save_eda_figures(df: pd.DataFrame, figure_dir: Path) -> dict[str, str]:
    ensure_project_directories()
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    apply_project_style()

    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["No", "Yes"]
    sns.countplot(
        data=df,
        x=TARGET_COLUMN,
        order=order,
        hue=TARGET_COLUMN,
        palette=[TEAL, RED],
        legend=False,
        ax=ax,
    )
    ax.set_title("Churn Class Distribution")
    ax.set_xlabel("Churn")
    output_path = figure_dir / "churn_class_distribution.png"
    add_bar_labels(ax, fmt="{:.0f}")
    polish_axes(ax)
    paths["class_distribution"] = save_chart(fig, output_path)

    churn_rate_by_contract = (
        df.assign(churn_flag=df[TARGET_COLUMN].map({"No": 0, "Yes": 1}))
        .groupby("Contract")["churn_flag"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=churn_rate_by_contract,
        x="Contract",
        y="churn_flag",
        hue="Contract",
        palette=sns.color_palette(WARM_PALETTE, n_colors=len(churn_rate_by_contract)),
        legend=False,
        ax=ax,
    )
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract")
    ax.set_ylabel("Churn rate")
    output_path = figure_dir / "churn_rate_by_contract.png"
    add_bar_labels(ax, percent=True)
    polish_axes(ax, rotate_x=18, percent_y=True)
    paths["churn_rate_by_contract"] = save_chart(fig, output_path)

    plot_df = df.copy()
    plot_df["TotalCharges"] = pd.to_numeric(plot_df["TotalCharges"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=plot_df,
        x="Contract",
        y="MonthlyCharges",
        hue=TARGET_COLUMN,
        palette=[TEAL, RED],
        ax=ax,
    )
    ax.set_title("Monthly Charges by Contract and Churn")
    output_path = figure_dir / "churn_contract_monthlycharges.png"
    polish_axes(ax, rotate_x=18)
    paths["contract_vs_monthlycharges"] = save_chart(fig, output_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=plot_df,
        x="MonthlyCharges",
        hue=TARGET_COLUMN,
        bins=30,
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
        palette=[TEAL, RED],
        ax=ax,
    )
    ax.set_title("Monthly Charges Distribution by Churn")
    ax.set_xlabel("Monthly charges")
    output_path = figure_dir / "churn_monthlycharges_distribution.png"
    polish_axes(ax)
    paths["monthlycharges_distribution"] = save_chart(fig, output_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(
        data=plot_df,
        x="tenure",
        hue=TARGET_COLUMN,
        fill=True,
        common_norm=False,
        palette=[TEAL, RED],
        ax=ax,
    )
    ax.set_title("Tenure Distribution by Churn")
    output_path = figure_dir / "churn_tenure_density.png"
    polish_axes(ax)
    paths["tenure_density"] = save_chart(fig, output_path)

    churn_rate_by_payment = (
        df.assign(churn_flag=df[TARGET_COLUMN].map({"No": 0, "Yes": 1}))
        .groupby("PaymentMethod")["churn_flag"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=churn_rate_by_payment,
        y="PaymentMethod",
        x="churn_flag",
        hue="PaymentMethod",
        palette=sns.color_palette(SEQUENTIAL_PALETTE, n_colors=len(churn_rate_by_payment)),
        legend=False,
        ax=ax,
    )
    ax.set_title("Churn Rate by Payment Method")
    ax.set_xlabel("Churn rate")
    ax.set_ylabel("")
    output_path = figure_dir / "churn_rate_by_payment_method.png"
    add_bar_labels(ax, percent=True)
    polish_axes(ax, percent_x=True)
    paths["churn_rate_by_payment_method"] = save_chart(fig, output_path)

    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    service_rates = []
    for column in service_columns:
        rate = (
            df.assign(churn_flag=df[TARGET_COLUMN].map({"No": 0, "Yes": 1}))
            .query(f"`{column}` == 'No'")["churn_flag"]
            .mean()
        )
        service_rates.append({"service_gap": f"No {column}", "churn_rate": rate})
    service_rate_df = pd.DataFrame(service_rates).sort_values(
        by="churn_rate", ascending=False
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=service_rate_df,
        y="service_gap",
        x="churn_rate",
        hue="service_gap",
        palette=sns.color_palette(SEQUENTIAL_PALETTE, n_colors=len(service_rate_df)),
        legend=False,
        ax=ax,
    )
    ax.set_title("Churn Rate for Customers Missing Add-on Services")
    ax.set_xlabel("Churn rate")
    ax.set_ylabel("")
    output_path = figure_dir / "churn_rate_by_missing_services.png"
    add_bar_labels(ax, percent=True)
    polish_axes(ax, percent_x=True)
    paths["churn_rate_by_missing_services"] = save_chart(fig, output_path)

    return paths


def save_model_figures(
    cv_results: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    best_threshold: float,
    feature_importance: pd.DataFrame,
    figure_dir: Path,
) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    comparison = cv_results.sort_values("cv_f1_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=comparison,
        y="model",
        x="cv_f1_mean",
        hue="model",
        palette=sns.color_palette(SEQUENTIAL_PALETTE, n_colors=len(comparison)),
        legend=False,
        ax=ax,
    )
    ax.set_title("Cross-Validation F1-Score by Model")
    ax.set_xlabel("CV F1-score")
    ax.set_ylabel("")
    best_f1 = comparison["cv_f1_mean"].max()
    ax.axvline(best_f1, color=GREEN, linestyle="--", linewidth=1.8, label="Best F1")
    ax.legend(loc="lower right")
    output_path = figure_dir / "churn_model_comparison_f1.png"
    add_bar_labels(ax, fmt="{:.3f}")
    polish_axes(ax)
    paths["model_comparison_f1"] = save_chart(fig, output_path)

    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="crest",
        linewidths=1,
        linecolor="#f7f4ee",
        cbar=False,
        ax=ax,
    )
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    output_path = figure_dir / "churn_confusion_matrix.png"
    polish_axes(ax)
    paths["confusion_matrix"] = save_chart(fig, output_path)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.5, label=f"ROC curve (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=RED, linewidth=1.8, label="Random baseline")
    ax.set_title("ROC curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    output_path = figure_dir / "churn_roc_curve.png"
    polish_axes(ax)
    paths["roc_curve"] = save_chart(fig, output_path)

    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=TEAL, linewidth=2.5)
    baseline = float(np.mean(y_test))
    ax.axhline(
        baseline,
        linestyle="--",
        color=RED,
        linewidth=1.8,
        label=f"Churn baseline={baseline:.2f}",
    )
    ax.set_title("Precision-Recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    output_path = figure_dir / "churn_precision_recall_curve.png"
    polish_axes(ax)
    paths["precision_recall_curve"] = save_chart(fig, output_path)

    threshold_scores = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )
    threshold_scores["f1"] = (
        2
        * threshold_scores["precision"]
        * threshold_scores["recall"]
        / (threshold_scores["precision"] + threshold_scores["recall"] + 1e-12)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        threshold_scores["threshold"],
        threshold_scores["f1"],
        color=GREEN,
        linewidth=2.4,
        label="F1-score",
    )
    ax.axvline(best_threshold, linestyle="--", color=RED, linewidth=2, label="Best threshold")
    ax.set_title("F1-Score by Decision Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1-score")
    ax.legend()
    output_path = figure_dir / "churn_threshold_vs_f1.png"
    polish_axes(ax)
    paths["threshold_vs_f1"] = save_chart(fig, output_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=feature_importance,
        y="feature",
        x="importance",
        hue="feature",
        palette=sns.color_palette(SEQUENTIAL_PALETTE, n_colors=len(feature_importance)),
        legend=False,
        ax=ax,
    )
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    output_path = figure_dir / "churn_feature_importance.png"
    add_bar_labels(ax, fmt="{:.2f}")
    polish_axes(ax)
    paths["feature_importance"] = save_chart(fig, output_path)

    return paths


def build_error_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    analysis = X_test.copy()
    analysis["actual"] = y_test.to_numpy()
    analysis["predicted"] = predictions
    analysis["predicted_probability"] = probabilities
    analysis["is_error"] = (analysis["actual"] != analysis["predicted"]).astype(int)
    grouped = (
        analysis.groupby("Contract")
        .agg(
            sample_count=("is_error", "size"),
            actual_churn_rate=("actual", "mean"),
            predicted_churn_rate=("predicted", "mean"),
            mean_predicted_probability=("predicted_probability", "mean"),
            error_rate=("is_error", "mean"),
        )
        .sort_values(by="error_rate", ascending=False)
        .reset_index()
    )
    return grouped


def classification_metric_row(
    stage: str,
    model_name: str,
    threshold: float,
    y_test: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | str]:
    return {
        "stage": stage,
        "model": model_name,
        "decision_threshold": threshold,
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_precision": precision_score(y_test, predictions, zero_division=0),
        "test_recall": recall_score(y_test, predictions, zero_division=0),
        "test_f1": f1_score(y_test, predictions, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, probabilities),
    }


def build_optimization_comparison(
    best_model_name: str,
    y_test: pd.Series,
    probabilities: np.ndarray,
    best_threshold: float,
) -> pd.DataFrame:
    default_predictions = (probabilities >= 0.5).astype(int)
    tuned_predictions = (probabilities >= best_threshold).astype(int)
    return pd.DataFrame(
        [
            classification_metric_row(
                "before_threshold_tuning",
                best_model_name,
                0.5,
                y_test,
                probabilities,
                default_predictions,
            ),
            classification_metric_row(
                "after_threshold_tuning",
                best_model_name,
                best_threshold,
                y_test,
                probabilities,
                tuned_predictions,
            ),
        ]
    )


def run_analysis() -> ClassificationArtifacts:
    ensure_project_directories()
    raw_df = load_churn_data()
    eda_figures = save_eda_figures(raw_df, CHURN_FIGURE_DIR)

    X, y = prepare_dataset(raw_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv_results = evaluate_models(X_train, y_train)
    best_model_name = cv_results.iloc[0]["model"]
    search = tune_best_model(best_model_name, X_train, y_train)
    best_pipeline = search.best_estimator_
    best_threshold = find_best_threshold(best_pipeline, X_train, y_train)
    best_pipeline.fit(X_train, y_train)
    probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= best_threshold).astype(int)

    metrics_df = pd.DataFrame(
        [
            {
                "model": best_model_name,
                "decision_threshold": best_threshold,
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision": precision_score(y_test, predictions, zero_division=0),
                "test_recall": recall_score(y_test, predictions, zero_division=0),
                "test_f1": f1_score(y_test, predictions, zero_division=0),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
            }
        ]
    )
    optimization_comparison = build_optimization_comparison(
        best_model_name,
        y_test,
        probabilities,
        best_threshold,
    )

    report_df = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            target_names=["No churn", "Churn"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    feature_importance = extract_feature_importance(search)
    error_analysis = build_error_analysis(
        X_test,
        y_test,
        predictions,
        probabilities,
    )
    model_figures = save_model_figures(
        cv_results,
        y_test,
        probabilities,
        predictions,
        best_threshold,
        feature_importance,
        CHURN_FIGURE_DIR,
    )

    cv_results.to_csv(CHURN_TABLE_DIR / "cv_results.csv", index=False)
    metrics_df.to_csv(CHURN_TABLE_DIR / "test_metrics.csv", index=False)
    optimization_comparison.to_csv(
        CHURN_TABLE_DIR / "optimization_comparison.csv", index=False
    )
    report_df.to_csv(CHURN_TABLE_DIR / "classification_report.csv")
    feature_importance.to_csv(CHURN_TABLE_DIR / "feature_importance.csv", index=False)
    error_analysis.to_csv(CHURN_TABLE_DIR / "error_analysis_by_contract.csv", index=False)

    return ClassificationArtifacts(
        cv_results=cv_results,
        tuned_metrics=metrics_df,
        optimization_comparison=optimization_comparison,
        feature_importance=feature_importance,
        classification_report_df=report_df,
        error_analysis=error_analysis,
        best_model_name=best_model_name,
        best_params=search.best_params_,
        best_threshold=best_threshold,
        figures={**eda_figures, **model_figures},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the churn classification project.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show generated charts after the analysis finishes.",
    )
    args = parser.parse_args()

    results = run_analysis()
    print("Churn classification best model:", results.best_model_name)
    print(results.tuned_metrics.to_string(index=False))

    if args.show:
        from .figure_viewer import show_saved_figures

        show_saved_figures(results.figures, title_prefix="Churn")


if __name__ == "__main__":
    main()
