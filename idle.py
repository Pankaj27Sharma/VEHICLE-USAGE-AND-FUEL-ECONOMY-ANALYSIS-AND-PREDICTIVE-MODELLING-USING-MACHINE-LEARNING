# ============================================================
# VEHICLE USAGE AND FUEL ECONOMY ANALYSIS
# USING MACHINE LEARNING
# Dataset: 2021 VIUS
# ============================================================

import os
import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
OUTPUT = "project_outputs"
os.makedirs(OUTPUT, exist_ok=True)


# ============================================================
# FUNCTION TO SAVE GRAPHS
# ============================================================

def save_plot(filename):
    plt.tight_layout()
    path = os.path.join(OUTPUT, filename)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Graph saved:", filename)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n========== 1. DATA COLLECTION ==========")

df = pd.read_csv(
    "vius_2021_puf.csv",
    low_memory=False
)

print("Dataset Shape:", df.shape)
print("\nFirst Five Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all").T)


# ============================================================
# 2. DATA QUALITY ANALYSIS
# ============================================================

print("\n========== 2. DATA QUALITY ANALYSIS ==========")

print("Duplicate Records:", df.duplicated().sum())

missing = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Missing Percentage": df.isnull().mean() * 100
})

missing = missing.sort_values(
    "Missing Percentage",
    ascending=False
)

print("\nMissing Value Analysis:")
print(missing.head(25))

missing.to_csv(
    os.path.join(OUTPUT, "missing_value_analysis.csv")
)

plt.figure(figsize=(11, 6))

missing.head(20)["Missing Percentage"].plot(kind="bar")

plt.title("Missing Values by Attributes")
plt.xlabel("Attributes")
plt.ylabel("Missing Percentage")
plt.xticks(rotation=65)

save_plot("01_missing_values.png")


# ============================================================
# 3. REGRESSION DATA PREPARATION
# ============================================================

print("\n========== 3. DATA PREPROCESSING ==========")

reg_target = "MPG"

reg_categorical = [
    "VEHTYPE",
    "FUELTYPE",
    "GVWR_CLASS",
    "TRANSMISSION",
    "CYLINDERS",
    "NUMGEARS",
    "DRIVEAXLES",
    "CUBICINCHDISP",
    "MODELYEAR"
]

reg_numeric = ["MILESANNL"]

required = (
    [reg_target] +
    reg_categorical +
    reg_numeric
)

missing_columns = [
    col for col in required
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing columns: " + str(missing_columns)
    )

data = df[required].copy()

data["MPG"] = pd.to_numeric(
    data["MPG"].replace(["X", ""], np.nan),
    errors="coerce"
)

data["MILESANNL"] = pd.to_numeric(
    data["MILESANNL"].replace(["X", ""], np.nan),
    errors="coerce"
)

for col in reg_categorical:
    data[col] = data[col].replace(
        ["X", ""], np.nan
    )

data = data.dropna(subset=["MPG"])

data = data[
    (data["MPG"] > 0) &
    (data["MPG"] <= 100)
].copy()

print("Original Records:", len(df))
print("Records Used:", len(data))

print("\nMissing Values:")
print(data.isnull().sum())


# ============================================================
# 4. OUTLIER DETECTION USING IQR
# ============================================================

print("\n========== 4. OUTLIER ANALYSIS ==========")

Q1 = data["MPG"].quantile(0.25)
Q3 = data["MPG"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = data[
    (data["MPG"] < lower_limit) |
    (data["MPG"] > upper_limit)
]

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)
print("Potential Outliers:", len(outliers))

# Retain potential outliers because they may be valid observations.

plt.figure(figsize=(8, 5))
plt.boxplot(data["MPG"].dropna())

plt.title("MPG Outlier Detection")
plt.ylabel("MPG")

save_plot("02_mpg_outliers.png")


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ==============================================================

print("\n========== 5. EXPLORATORY DATA ANALYSIS ==========")

# MPG Distribution

plt.figure(figsize=(9, 5))

sns.histplot(
    data["MPG"],
    bins=30,
    kde=True
)

plt.title("Distribution of Vehicle Fuel Economy")
plt.xlabel("MPG")
plt.ylabel("Frequency")

save_plot("03_mpg_distribution.png")


# Vehicle Type Distribution

plt.figure(figsize=(9, 5))

data["VEHTYPE"].value_counts().plot(kind="bar")

plt.title("Vehicle Type Distribution")
plt.xlabel("Vehicle Type Code")
plt.ylabel("Number of Vehicles")

save_plot("04_vehicle_distribution.png")


# Fuel Type Distribution

plt.figure(figsize=(9, 5))

data["FUELTYPE"].value_counts().head(12).plot(kind="bar")

plt.title("Fuel Type Distribution")
plt.xlabel("Fuel Type Code")
plt.ylabel("Number of Vehicles")

save_plot("05_fuel_distribution.png")


# MPG by Vehicle Type

plt.figure(figsize=(10, 5))

sns.boxplot(
    data=data,
    x="VEHTYPE",
    y="MPG"
)

plt.title("MPG by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("MPG")

save_plot("06_mpg_by_vehicle_type.png")


# MPG by Fuel Type

fuel_order = (
    data.groupby("FUELTYPE")["MPG"]
    .median()
    .sort_values(ascending=False)
    .index
)

plt.figure(figsize=(11, 5))

sns.boxplot(
    data=data,
    x="FUELTYPE",
    y="MPG",
    order=fuel_order
)

plt.title("MPG by Fuel Type")
plt.xlabel("Fuel Type Code")
plt.ylabel("MPG")
plt.xticks(rotation=45)

save_plot("07_mpg_by_fuel_type.png")


# Annual Mileage Distribution

plt.figure(figsize=(9, 5))

sns.histplot(
    data["MILESANNL"].dropna(),
    bins=40,
    kde=True
)

plt.title("Annual Mileage Distribution")
plt.xlabel("Annual Miles")
plt.ylabel("Frequency")

save_plot("08_annual_mileage.png")


# Correlation Heatmap

plt.figure(figsize=(7, 5))

correlation = data[
    ["MPG", "MILESANNL"]
].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

save_plot("09_correlation_heatmap.png")


# ============================================================
# 6. REGRESSION PREPROCESSING
# ============================================================

print("\n========== 6. REGRESSION PREPROCESSING ==========")

X = data[reg_categorical + reg_numeric]
y = data["MPG"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

reg_preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, reg_numeric),
    ("categorical", categorical_pipeline, reg_categorical)
])

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))


# ============================================================
# 7. FIVE REGRESSION MODELS
# ============================================================

print("\n========== 7. REGRESSION MODELS ==========")

reg_models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=18,
        min_samples_leaf=8,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=120,
        max_depth=22,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=120,
        max_depth=22,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42
    ),

    "Linear SVR": LinearSVR(
        C=0.5,
        epsilon=0.1,
        max_iter=12000,
        random_state=42
    )
}

reg_results = []
reg_predictions = {}
fitted_reg_models = {}

for name, model in reg_models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("preprocessor", reg_preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    prediction = pipeline.predict(X_test)

    fitted_reg_models[name] = pipeline
    reg_predictions[name] = prediction

    mae = mean_absolute_error(y_test, prediction)

    rmse = np.sqrt(
        mean_squared_error(y_test, prediction)
    )

    r2 = r2_score(y_test, prediction)

    reg_results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))


reg_results_df = pd.DataFrame(
    reg_results
).sort_values("RMSE")

print("\nREGRESSION COMPARISON")
print(reg_results_df.to_string(index=False))

reg_results_df.to_csv(
    os.path.join(OUTPUT, "regression_results.csv"),
    index=False
)


# ============================================================
# 8. REGRESSION COMPARISON GRAPHS
# ============================================================

plt.figure(figsize=(10, 5))

sns.barplot(
    data=reg_results_df,
    x="Model",
    y="RMSE"
)

plt.title("Regression RMSE Comparison")
plt.xticks(rotation=30)

save_plot("10_regression_rmse.png")


plt.figure(figsize=(10, 5))

sns.barplot(
    data=reg_results_df,
    x="Model",
    y="R2 Score"
)

plt.title("Regression R2 Score Comparison")
plt.xticks(rotation=30)

save_plot("11_regression_r2.png")


# ============================================================
# 9. CROSS VALIDATION
# ============================================================

print("\n========== 9. CROSS VALIDATION ==========")

cv_size = min(8000, len(X_train))

cv_indices = X_train.sample(
    n=cv_size,
    random_state=42
).index

X_cv = X_train.loc[cv_indices]
y_cv = y_train.loc[cv_indices]

cv_results = []

for name in [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "Extra Trees"
]:

    print("Cross Validating:", name)

    pipeline = Pipeline([
        ("preprocessor", reg_preprocessor),
        ("model", reg_models[name])
    ])

    scores = cross_val_score(
        pipeline,
        X_cv,
        y_cv,
        cv=3,
        scoring="r2",
        n_jobs=1
    )

    cv_results.append({
        "Model": name,
        "Mean CV R2": scores.mean(),
        "Standard Deviation": scores.std()
    })

cv_df = pd.DataFrame(cv_results)

print("\nCROSS VALIDATION RESULTS")
print(cv_df.to_string(index=False))

cv_df.to_csv(
    os.path.join(OUTPUT, "cross_validation_results.csv"),
    index=False
)


# ============================================================
# 10. BEST REGRESSION MODEL ANALYSIS
# ============================================================

print("\n========== 10. BEST REGRESSION MODEL ==========")

best_reg_name = reg_results_df.iloc[0]["Model"]

best_prediction = reg_predictions[best_reg_name]

print("Best Regression Model:", best_reg_name)


# Actual vs Predicted MPG

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    best_prediction,
    alpha=0.35
)

low = min(y_test.min(), best_prediction.min())
high = max(y_test.max(), best_prediction.max())

plt.plot(
    [low, high],
    [low, high],
    "r--"
)

plt.title("Actual vs Predicted MPG")
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")

save_plot("12_actual_vs_predicted.png")


# Residual Analysis

residuals = y_test - best_prediction

plt.figure(figsize=(8, 5))

sns.histplot(
    residuals,
    bins=35,
    kde=True
)

plt.title("Residual Distribution")
plt.xlabel("Prediction Error")

save_plot("13_residual_analysis.png")


# Feature Importance

rf_pipeline = fitted_reg_models["Random Forest"]

feature_names = (
    rf_pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    rf_pipeline.named_steps["model"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(
    "Importance",
    ascending=False
)

print("\nTOP 15 IMPORTANT FEATURES")
print(importance_df.head(15).to_string(index=False))

importance_df.to_csv(
    os.path.join(OUTPUT, "feature_importance.csv"),
    index=False
)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance_df.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Random Forest Features")

save_plot("14_feature_importance.png")


# Save predictions

pd.DataFrame({
    "Actual MPG": y_test.values,
    "Predicted MPG": best_prediction
}).to_csv(
    os.path.join(OUTPUT, "best_regression_predictions.csv"),
    index=False
)


# ============================================================
# 11. CLASSIFICATION DATASET
# ============================================================

print("\n========== 11. CLASSIFICATION DATA ==========")

class_target = "VEHTYPE"

class_categorical = [
    "FUELTYPE",
    "GVWR_CLASS",
    "TRANSMISSION",
    "CYLINDERS",
    "NUMGEARS",
    "DRIVEAXLES",
    "CUBICINCHDISP",
    "MODELYEAR"
]

class_numeric = ["MILESANNL"]

class_required = (
    [class_target] +
    class_categorical +
    class_numeric
)

class_data = df[class_required].copy()

for col in class_categorical + [class_target]:

    class_data[col] = class_data[col].replace(
        ["X", ""], np.nan
    )

class_data["MILESANNL"] = pd.to_numeric(
    class_data["MILESANNL"].replace(["X", ""], np.nan),
    errors="coerce"
)

class_data = class_data.dropna(
    subset=[class_target]
)

class_counts = class_data[class_target].value_counts()

valid_classes = class_counts[
    class_counts >= 10
].index

class_data = class_data[
    class_data[class_target].isin(valid_classes)
].copy()

print("Classification Records:", len(class_data))
print("\nVehicle Type Distribution:")
print(class_data[class_target].value_counts())

Xc = class_data[
    class_categorical + class_numeric
]

yc = class_data[class_target]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc,
    yc,
    test_size=0.20,
    random_state=42,
    stratify=yc
)

class_preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, class_numeric),
    ("categorical", categorical_pipeline, class_categorical)
])


# ============================================================
# 12. FOUR CLASSIFICATION MODELS
# ============================================================

print("\n========== 12. CLASSIFICATION MODELS ==========")

class_models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=120,
        max_depth=20,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )
}

class_results = []
class_predictions = {}

for name, model in class_models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("preprocessor", class_preprocessor),
        ("model", model)
    ])

    pipeline.fit(Xc_train, yc_train)

    prediction = pipeline.predict(Xc_test)

    class_predictions[name] = prediction

    accuracy = accuracy_score(
        yc_test,
        prediction
    )

    precision = precision_score(
        yc_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        yc_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        yc_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    class_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))


class_results_df = pd.DataFrame(
    class_results
).sort_values(
    "F1 Score",
    ascending=False
)

print("\nCLASSIFICATION COMPARISON")
print(class_results_df.to_string(index=False))

class_results_df.to_csv(
    os.path.join(OUTPUT, "classification_results.csv"),
    index=False
)


# ============================================================
# 13. CLASSIFICATION COMPARISON GRAPHS
# ============================================================

plt.figure(figsize=(10, 5))

sns.barplot(
    data=class_results_df,
    x="Model",
    y="Accuracy"
)

plt.title("Classification Accuracy Comparison")
plt.xticks(rotation=25)

save_plot("15_classification_accuracy.png")


plt.figure(figsize=(10, 5))

sns.barplot(
    data=class_results_df,
    x="Model",
    y="F1 Score"
)

plt.title("Classification F1 Score Comparison")
plt.xticks(rotation=25)

save_plot("16_classification_f1.png")


# ============================================================
# 14. CONFUSION MATRICES
# ============================================================

print("\n========== 14. CONFUSION MATRICES ==========")

best_class_name = class_results_df.iloc[0]["Model"]

print("Best Classifier:", best_class_name)

for name, prediction in class_predictions.items():

    print("\nClassification Report:", name)

    print(
        classification_report(
            yc_test,
            prediction,
            zero_division=0
        )
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay.from_predictions(
        yc_test,
        prediction,
        ax=ax,
        xticks_rotation=45,
        cmap="Blues"
    )

    ax.set_title("Confusion Matrix - " + name)

    fig.tight_layout()

    filename = (
        "confusion_" +
        name.replace(" ", "_") +
        ".png"
    )

    fig.savefig(
        os.path.join(OUTPUT, filename),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", filename)


# ============================================================
# 15. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("FINAL PROJECT SUMMARY")
print("=" * 55)

print("Original Dataset Records:", len(df))
print("Regression Records:", len(data))
print("Classification Records:", len(class_data))

print("\nBest Regression Model:", best_reg_name)

print(
    "Best Regression RMSE:",
    round(reg_results_df.iloc[0]["RMSE"], 4)
)

print("\nBest Classification Model:", best_class_name)

print(
    "Best Classification F1:",
    round(class_results_df.iloc[0]["F1 Score"], 4)
)

print("\nREGRESSION RESULTS:")
print(reg_results_df.to_string(index=False))

print("\nCLASSIFICATION RESULTS:")
print(class_results_df.to_string(index=False))

print("\nAll graphs and results saved in:", OUTPUT)

print("\nPROJECT EXECUTION COMPLETED SUCCESSFULLY!")


# ============================================================
# IMPORTANT RESEARCH NOTES
# ============================================================
#
# 1. Dataset: 2021 U.S. Vehicle Inventory and Use Survey.
#
# 2. MPG values that are missing or marked X are excluded
#    from regression target training.
#
# 3. Missing predictor values are imputed inside pipelines.
#
# 4. IQR outliers are identified but retained because they
#    may represent genuine vehicle observations.
#
# 5. Classification predicts coded VEHTYPE categories.
#    Consult the official VIUS data dictionary before
#    interpreting category codes.
#
# 6. Survey weights are not applied. Results are an
#    unweighted predictive modelling exercise.
#
# 7. Regression and classification are separate experiments.
#
# ============================================================
