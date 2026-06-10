"""CO2 regression project.

Run with:
    python -m ml_coursework.co2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .settings import (
    CO2_FIGURE_DIR,
    CO2_TABLE_DIR,
    CV_FOLDS,
    RANDOM_STATE,
    TEST_SIZE,
    TOP_FEATURES_TO_SHOW,
)
from .data_loader import ensure_project_directories, load_co2_data
from .visual_style import (
    AMBER,
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


TARGET_COLUMN = "CO2EMISSIONS"
DROP_COLUMNS = ["MODELYEAR", "MODEL"]
NUMERIC_FEATURES = [
    "ENGINESIZE",
    "CYLINDERS",
    "FUELCONSUMPTION_CITY",
    "FUELCONSUMPTION_HWY",
    "FUELCONSUMPTION_COMB",
    "FUELCONSUMPTION_COMB_MPG",
]
CATEGORICAL_FEATURES = ["MAKE", "VEHICLECLASS", "TRANSMISSION", "FUELTYPE"]


@dataclass
class RegressionArtifacts:
    cv_results: pd.DataFrame
    tuned_metrics: pd.DataFrame
    optimization_comparison: pd.DataFrame
    feature_importance: pd.DataFrame
    error_analysis: pd.DataFrame
    best_model_name: str
    best_params: dict
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
        "Dummy Regressor": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def prepare_dataset(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = dataframe.copy()
    df = df.drop(columns=DROP_COLUMNS)
    df = df.drop_duplicates().reset_index(drop=True)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]
    return X, y


def evaluate_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    results = []
    preprocessor = build_preprocessor()

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
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=-1,
        )
        results.append(
            {
                "model": model_name,
                "cv_rmse_mean": -scores["test_rmse"].mean(),
                "cv_rmse_std": scores["test_rmse"].std(),
                "cv_mae_mean": -scores["test_mae"].mean(),
                "cv_r2_mean": scores["test_r2"].mean(),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by="cv_rmse_mean", ascending=True
    )
    return results_df.reset_index(drop=True)


def tune_best_model(
    model_name: str, X_train: pd.DataFrame, y_train: pd.Series
) -> RandomizedSearchCV:
    preprocessor = build_preprocessor()
    search_spaces = {
        "Random Forest": {
            "model__n_estimators": [200, 300, 500, 700],
            "model__max_depth": [None, 8, 12, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.8, None],
        },
        "Gradient Boosting": {
            "model__n_estimators": [150, 250, 350],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0],
            "model__min_samples_leaf": [1, 3, 5],
        },
        "Linear Regression": {},
    }

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", build_models()[model_name])]
    )
    param_distributions = search_spaces[model_name]

    if not param_distributions:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions={"model__fit_intercept": [True]},
            n_iter=1,
            cv=CV_FOLDS,
            scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
    else:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=12,
            cv=CV_FOLDS,
            scoring="neg_root_mean_squared_error",
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
        raw_values = np.abs(model.coef_)

    importance = pd.DataFrame(
        {"feature": feature_names, "importance": raw_values}
    ).sort_values(by="importance", ascending=False)
    return importance.head(TOP_FEATURES_TO_SHOW).reset_index(drop=True)


def save_eda_figures(df: pd.DataFrame, figure_dir: Path) -> dict[str, str]:
    ensure_project_directories()
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    apply_project_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN], kde=True, color=TEAL, edgecolor="white", ax=ax)
    median_emissions = df[TARGET_COLUMN].median()
    ax.axvline(
        median_emissions,
        color=RED,
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_emissions:.0f} g/km",
    )
    ax.set_title("Distribution of CO2 Emissions")
    ax.set_xlabel("CO2 emissions (g/km)")
    ax.legend(loc="upper right")
    output_path = figure_dir / "co2_target_distribution.png"
    polish_axes(ax)
    paths["target_distribution"] = save_chart(fig, output_path)

    corr_columns = NUMERIC_FEATURES + [TARGET_COLUMN]
    corr_labels = {
        "ENGINESIZE": "Engine",
        "CYLINDERS": "Cylinders",
        "FUELCONSUMPTION_CITY": "City FC",
        "FUELCONSUMPTION_HWY": "Highway FC",
        "FUELCONSUMPTION_COMB": "Combined FC",
        "FUELCONSUMPTION_COMB_MPG": "Combined MPG",
        TARGET_COLUMN: "CO2",
    }
    corr_matrix = df[corr_columns].corr().rename(index=corr_labels, columns=corr_labels)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="crest",
        linewidths=0.7,
        linecolor="#f7f4ee",
        square=True,
        cbar_kws={"shrink": 0.78},
        ax=ax,
    )
    ax.set_title("Correlation Between Numeric Features and CO2")
    output_path = figure_dir / "co2_correlation_heatmap.png"
    polish_axes(ax, rotate_x=35)
    paths["correlation_heatmap"] = save_chart(fig, output_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="FUELCONSUMPTION_COMB",
        y=TARGET_COLUMN,
        hue="FUELTYPE",
        palette=sns.color_palette(QUALITATIVE_PALETTE, n_colors=df["FUELTYPE"].nunique()),
        alpha=0.75,
        s=56,
        edgecolor="white",
        linewidth=0.25,
        ax=ax,
    )
    ax.set_title("CO2 vs Combined Fuel Consumption")
    ax.set_xlabel("Fuel consumption combined")
    ax.set_ylabel("CO2 emissions (g/km)")
    output_path = figure_dir / "co2_fuel_consumption_scatter.png"
    polish_axes(ax)
    paths["fuel_consumption_scatter"] = save_chart(fig, output_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        data=df,
        x="ENGINESIZE",
        y=TARGET_COLUMN,
        color=BLUE,
        scatter_kws={"alpha": 0.52, "s": 44, "edgecolor": "white", "linewidths": 0.25},
        line_kws={"color": RED, "linewidth": 2.5},
        ax=ax,
    )
    ax.set_title("Engine Size vs CO2 Emissions")
    ax.set_xlabel("Engine size")
    ax.set_ylabel("CO2 emissions (g/km)")
    output_path = figure_dir / "co2_engine_size_regression.png"
    polish_axes(ax)
    paths["engine_size_regression"] = save_chart(fig, output_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    ordered_classes = (
        df.groupby("VEHICLECLASS")[TARGET_COLUMN].median().sort_values().index
    )
    sns.boxplot(
        data=df,
        x="VEHICLECLASS",
        y=TARGET_COLUMN,
        order=ordered_classes,
        hue="VEHICLECLASS",
        palette=sns.color_palette(SEQUENTIAL_PALETTE, n_colors=len(ordered_classes)),
        legend=False,
        ax=ax,
    )
    ax.set_title("CO2 Emissions by Vehicle Class")
    ax.set_xlabel("")
    ax.set_ylabel("CO2 emissions (g/km)")
    output_path = figure_dir / "co2_by_vehicle_class.png"
    polish_axes(ax, rotate_x=70)
    paths["vehicle_class_boxplot"] = save_chart(fig, output_path)

    top_makes = (
        df.groupby("MAKE")[TARGET_COLUMN]
        .mean()
        .sort_values(ascending=False)
        .head(12)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=top_makes,
        y="MAKE",
        x=TARGET_COLUMN,
        hue="MAKE",
        palette=sns.color_palette(WARM_PALETTE, n_colors=len(top_makes)),
        legend=False,
        ax=ax,
    )
    ax.set_title("Top Makes by Mean CO2 Emissions")
    ax.set_xlabel("Mean CO2 emissions (g/km)")
    ax.set_ylabel("")
    output_path = figure_dir / "co2_top_makes_mean_emissions.png"
    add_bar_labels(ax, fmt="{:.0f}")
    polish_axes(ax)
    paths["top_makes_mean_emissions"] = save_chart(fig, output_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="FUELTYPE",
        y=TARGET_COLUMN,
        hue="FUELTYPE",
        palette=sns.color_palette(QUALITATIVE_PALETTE, n_colors=df["FUELTYPE"].nunique()),
        legend=False,
        ax=ax,
    )
    ax.set_title("CO2 Emissions by Fuel Type")
    ax.set_xlabel("Fuel type")
    ax.set_ylabel("CO2 emissions (g/km)")
    output_path = figure_dir / "co2_by_fuel_type.png"
    polish_axes(ax)
    paths["fuel_type_boxplot"] = save_chart(fig, output_path)

    return paths


def save_model_figures(
    cv_results: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    residual_frame: pd.DataFrame,
    feature_importance: pd.DataFrame,
    figure_dir: Path,
) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    comparison = cv_results.sort_values("cv_rmse_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=comparison,
        y="model",
        x="cv_rmse_mean",
        hue="model",
        palette=sns.color_palette(SEQUENTIAL_PALETTE, n_colors=len(comparison)),
        legend=False,
        ax=ax,
    )
    ax.set_title("Cross-Validation RMSE by Model")
    ax.set_xlabel("CV RMSE")
    ax.set_ylabel("")
    best_rmse = comparison["cv_rmse_mean"].min()
    ax.axvline(best_rmse, color=GREEN, linestyle="--", linewidth=1.8, label="Best RMSE")
    ax.legend(loc="lower right")
    output_path = figure_dir / "co2_model_comparison_rmse.png"
    add_bar_labels(ax, fmt="{:.2f}")
    polish_axes(ax)
    paths["model_comparison_rmse"] = save_chart(fig, output_path)

    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(
        x=y_test,
        y=predictions,
        alpha=0.72,
        color=TEAL,
        edgecolor="white",
        linewidth=0.25,
        s=58,
        ax=ax,
    )
    min_axis = min(float(y_test.min()), float(predictions.min()))
    max_axis = max(float(y_test.max()), float(predictions.max()))
    ax.plot([min_axis, max_axis], [min_axis, max_axis], "--", color=RED, linewidth=2)
    ax.set_title("Actual vs Predicted CO2")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    output_path = figure_dir / "co2_actual_vs_predicted.png"
    polish_axes(ax)
    paths["actual_vs_predicted"] = save_chart(fig, output_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=residual_frame,
        x="predicted",
        y="residual",
        hue="FUELTYPE",
        palette=sns.color_palette(
            QUALITATIVE_PALETTE,
            n_colors=residual_frame["FUELTYPE"].nunique(),
        ),
        alpha=0.6,
        s=48,
        edgecolor="white",
        linewidth=0.2,
        ax=ax,
    )
    ax.axhline(0, linestyle="--", color=RED, linewidth=2)
    ax.set_title("Residuals vs Predicted CO2")
    ax.set_xlabel("Predicted CO2")
    ax.set_ylabel("Residual")
    output_path = figure_dir / "co2_residual_vs_predicted.png"
    polish_axes(ax)
    paths["residual_vs_predicted"] = save_chart(fig, output_path)

    residuals = y_test - predictions
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color=AMBER, edgecolor="white", ax=ax)
    ax.axvline(0, color=RED, linestyle="--", linewidth=2, label="Zero error")
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual")
    ax.legend(loc="upper right")
    output_path = figure_dir / "co2_residual_distribution.png"
    polish_axes(ax)
    paths["residual_distribution"] = save_chart(fig, output_path)

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
    output_path = figure_dir / "co2_feature_importance.png"
    add_bar_labels(ax, fmt="{:.2f}")
    polish_axes(ax)
    paths["feature_importance"] = save_chart(fig, output_path)

    return paths


def build_error_analysis(
    X_test: pd.DataFrame, y_test: pd.Series, predictions: np.ndarray
) -> pd.DataFrame:
    analysis = X_test.copy()
    analysis["actual"] = y_test.to_numpy()
    analysis["predicted"] = predictions
    analysis["absolute_error"] = np.abs(analysis["actual"] - analysis["predicted"])
    grouped = (
        analysis.groupby("VEHICLECLASS")
        .agg(
            sample_count=("absolute_error", "size"),
            mean_actual_co2=("actual", "mean"),
            mean_predicted_co2=("predicted", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
        )
        .sort_values(by="mean_absolute_error", ascending=False)
        .reset_index()
    )
    return grouped


def regression_metric_row(
    stage: str, model_name: str, y_test: pd.Series, predictions: np.ndarray
) -> dict[str, float | str]:
    return {
        "stage": stage,
        "model": model_name,
        "test_mae": mean_absolute_error(y_test, predictions),
        "test_rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "test_r2": r2_score(y_test, predictions),
    }


def build_optimization_comparison(
    best_model_name: str,
    tuned_pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    dummy_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    default_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", build_models()[best_model_name]),
        ]
    )

    dummy_pipeline.fit(X_train, y_train)
    default_pipeline.fit(X_train, y_train)

    rows = [
        regression_metric_row(
            "dummy_baseline",
            "Dummy Regressor",
            y_test,
            dummy_pipeline.predict(X_test),
        ),
        regression_metric_row(
            "before_tuning",
            best_model_name,
            y_test,
            default_pipeline.predict(X_test),
        ),
        regression_metric_row(
            "after_tuning",
            best_model_name,
            y_test,
            tuned_pipeline.predict(X_test),
        ),
    ]
    return pd.DataFrame(rows)


def run_analysis() -> RegressionArtifacts:
    ensure_project_directories()
    raw_df = load_co2_data()
    eda_figures = save_eda_figures(raw_df, CO2_FIGURE_DIR)

    X, y = prepare_dataset(raw_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    cv_results = evaluate_models(X_train, y_train)
    best_model_name = cv_results.iloc[0]["model"]
    search = tune_best_model(best_model_name, X_train, y_train)
    best_pipeline = search.best_estimator_
    predictions = best_pipeline.predict(X_test)
    residual_frame = X_test.copy()
    residual_frame["actual"] = y_test.to_numpy()
    residual_frame["predicted"] = predictions
    residual_frame["residual"] = residual_frame["actual"] - residual_frame["predicted"]

    metrics_df = pd.DataFrame(
        [
            {
                "model": best_model_name,
                "test_mae": mean_absolute_error(y_test, predictions),
                "test_rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "test_r2": r2_score(y_test, predictions),
            }
        ]
    )

    feature_importance = extract_feature_importance(search)
    error_analysis = build_error_analysis(X_test, y_test, predictions)
    optimization_comparison = build_optimization_comparison(
        best_model_name,
        best_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    model_figures = save_model_figures(
        cv_results,
        y_test,
        predictions,
        residual_frame,
        feature_importance,
        CO2_FIGURE_DIR,
    )

    cv_results.to_csv(CO2_TABLE_DIR / "cv_results.csv", index=False)
    metrics_df.to_csv(CO2_TABLE_DIR / "test_metrics.csv", index=False)
    feature_importance.to_csv(CO2_TABLE_DIR / "feature_importance.csv", index=False)
    error_analysis.to_csv(CO2_TABLE_DIR / "error_analysis_by_vehicleclass.csv", index=False)
    optimization_comparison.to_csv(
        CO2_TABLE_DIR / "optimization_comparison.csv", index=False
    )

    return RegressionArtifacts(
        cv_results=cv_results,
        tuned_metrics=metrics_df,
        optimization_comparison=optimization_comparison,
        feature_importance=feature_importance,
        error_analysis=error_analysis,
        best_model_name=best_model_name,
        best_params=search.best_params_,
        figures={**eda_figures, **model_figures},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CO2 regression project.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show generated charts after the analysis finishes.",
    )
    args = parser.parse_args()

    results = run_analysis()
    print("CO2 regression best model:", results.best_model_name)
    print(results.tuned_metrics.to_string(index=False))

    if args.show:
        from .figure_viewer import show_saved_figures

        show_saved_figures(results.figures, title_prefix="CO2")


if __name__ == "__main__":
    main()
