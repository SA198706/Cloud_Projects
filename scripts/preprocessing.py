"""
preprocessing.py
Sub-Objective 1.3: Data Pre-processing
- Summary statistics
- Missing value check
- Imputation (numeric columns -> median)
- Data type display
- Normalization (numeric columns -> StandardScaler)

Adapted for the REAL Lending Club "LoanStats" dataset (Kaggle:
adarshsng/lending-club-loan-data-csv or wordsforthewise/lending-club).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

NUMERIC_COLS = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
                'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
                'revol_bal', 'revol_util', 'total_acc']
CATEGORICAL_COLS = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
                     'verification_status', 'purpose']

# loan_status values that represent a resolved, known outcome.
# "Current", "In Grace Period", "Late (...)" are excluded: outcome not yet known.
DEFAULT_STATUSES = ['Charged Off', 'Default',
                     'Does not meet the credit policy. Status:Charged Off']
PAID_STATUSES = ['Fully Paid',
                  'Does not meet the credit policy. Status:Fully Paid']


def load_data(path='../data/loan_data_raw.csv'):
    df = pd.read_csv(path, low_memory=False)
    df = df[df['loan_status'].isin(DEFAULT_STATUSES + PAID_STATUSES)].copy()
    df['default'] = df['loan_status'].isin(DEFAULT_STATUSES).astype(int)
    # emp_length: "< 1 year" -> 0, "10+ years" -> 10, "3 years" -> 3, NaN stays NaN
    df['emp_length'] = (df['emp_length']
                         .str.extract(r'(\d+)')[0]
                         .astype(float))
    # term: "36 months" -> 36
    df['term'] = df['term'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    keep_cols = NUMERIC_COLS + CATEGORICAL_COLS + ['default']
    return df[[c for c in keep_cols if c in df.columns]]


def preprocess(df: pd.DataFrame) -> dict:
    result = {}

    # 1. Summary statistics
    result['summary_stats'] = df[NUMERIC_COLS].describe().to_dict()

    # 2. Data types
    result['dtypes'] = df.dtypes.astype(str).to_dict()

    # 3. Missing value check (before)
    missing_before = df.isnull().sum()
    result['missing_before'] = missing_before[missing_before > 0].to_dict()

    # 4. Impute numeric columns with median
    df_clean = df.copy()
    for col in NUMERIC_COLS:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Impute categorical with mode
    for col in CATEGORICAL_COLS:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    missing_after = df_clean.isnull().sum()
    result['missing_after'] = int(missing_after.sum())

    # 5. Normalize numeric columns (z-score) -> new *_scaled columns
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_clean[NUMERIC_COLS])
    for i, col in enumerate(NUMERIC_COLS):
        df_clean[f'{col}_scaled'] = scaled[:, i]

    result['rows'] = len(df_clean)
    result['default_rate'] = float(df_clean['default'].mean())
    result['df_clean'] = df_clean
    return result


if __name__ == '__main__':
    df = load_data()
    out = preprocess(df)
    print("Rows:", out['rows'])
    print("Missing values before imputation:\n", out['missing_before'])
    print("Missing values after imputation:", out['missing_after'])
    print("Default rate:", round(out['default_rate'], 4))
    print("\nDtypes:\n", out['dtypes'])
    out['df_clean'].to_csv('../data/loan_data_clean.csv', index=False)
    print("\nSaved cleaned dataset -> loan_data_clean.csv")