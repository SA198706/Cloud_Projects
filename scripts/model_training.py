"""
model_training.py

Sub-Objective 2.1 & 2.2

Model Preparation
-----------------
1. Load cleaned dataset from preprocessing.py
2. Encode categorical variables
3. Train/Test split (70/30)
4. Train two ML models
    - Logistic Regression
    - Random Forest
5. Save trained models
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocessing import (
    load_data,
    preprocess,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
)

# --------------------------------------------------------------------
# Directories
# --------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------
# Model names
# --------------------------------------------------------------------

LOGISTIC_MODEL = MODEL_DIR / "logistic_regression.pkl"
RANDOM_FOREST_MODEL = MODEL_DIR / "random_forest.pkl"

PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.pkl"

# --------------------------------------------------------------------
# Load processed data
# --------------------------------------------------------------------


def prepare_training_data():
    """
    Loads and preprocesses the Lending Club dataset.
    Drops the *_scaled columns added by preprocessing.py to avoid
    data leakage (those are normalised copies of the raw numeric
    columns already present in X; keeping both would inflate CV scores).
    """
    raw_df = load_data()
    result = preprocess(raw_df)
    df = result["df_clean"]
    # Drop StandardScaler duplicates (*_scaled) – the pipeline's own
    # ColumnTransformer will handle scaling internally.
    scaled_cols = [c for c in df.columns if c.endswith("_scaled")]
    df = df.drop(columns=scaled_cols)
    X = df.drop(columns=["default"])
    y = df["default"]
    return X, y


# --------------------------------------------------------------------
# Build preprocessing transformer
# --------------------------------------------------------------------


def create_preprocessor():
    """
    OneHotEncode categorical columns while keeping
    numeric columns unchanged.
    """

    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLS,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_COLS,
            ),
        ]
    )

    return transformer


# --------------------------------------------------------------------
# Split dataset
# --------------------------------------------------------------------


def split_dataset():

    X, y = prepare_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------
# Logistic Regression
# --------------------------------------------------------------------


def train_logistic_regression(
    X_train,
    y_train,
    preprocessor,
):
    """
    Train Logistic Regression Pipeline.
    """

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=10000,
                    solver='saga',
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, LOGISTIC_MODEL)

    return pipeline


# --------------------------------------------------------------------
# Random Forest
# --------------------------------------------------------------------


def train_random_forest(
    X_train,
    y_train,
    preprocessor,
):
    """
    Train Random Forest Pipeline.
    """

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, RANDOM_FOREST_MODEL)

    return pipeline


# --------------------------------------------------------------------
# Train both models
# --------------------------------------------------------------------


def train_models():
    """
    Main training function.
    """

    X_train, X_test, y_train, y_test = split_dataset()

    preprocessor = create_preprocessor()

    joblib.dump(preprocessor, PREPROCESSOR_FILE)

    logistic_model = train_logistic_regression(
        X_train,
        y_train,
        preprocessor,
    )

    random_forest_model = train_random_forest(
        X_train,
        y_train,
        preprocessor,
    )

    return {
        "logistic_model": logistic_model,
        "random_forest_model": random_forest_model,
        "X_test": X_test,
        "y_test": y_test,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "logistic_path": str(LOGISTIC_MODEL),
        "random_forest_path": str(RANDOM_FOREST_MODEL),
    }


# --------------------------------------------------------------------
# Standalone execution
# --------------------------------------------------------------------

if __name__ == "__main__":

    output = train_models()

    print("\n========== TRAINING COMPLETE ==========\n")

    print(f"Training Rows : {output['train_rows']}")
    print(f"Testing Rows  : {output['test_rows']}")

    print("\nModels Saved\n")

    print(output["logistic_path"])
    print(output["random_forest_path"])
