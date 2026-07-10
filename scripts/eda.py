"""
eda.py
Sub-Objective 1.4: Exploratory Data Analysis
- Correlation coefficients (numeric-numeric, categorical-target)
- Binning (dti, annual_inc into risk tiers)
- Encoding (one-hot for model-ready categoricals)
- Feature importance (via RandomForest)
- Visualizations (univariate + bivariate)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from preprocessing import load_data, preprocess, NUMERIC_COLS, CATEGORICAL_COLS

PLOTS_DIR = str(Path(__file__).resolve().parent.parent / "plots")


def run_eda(df_clean: pd.DataFrame) -> dict:
    result = {}

    # ---- 1. Correlation coefficients (numeric vs target) ----
    corr_matrix = df_clean[NUMERIC_COLS + ['default']].corr()
    result['correlation_with_target'] = corr_matrix['default'].drop('default').sort_values(
        key=abs, ascending=False).to_dict()

    # ---- 2. Binning ----
    df_clean['dti_tier'] = pd.cut(df_clean['dti'], bins=[-1, 10, 20, 30, 100],
                                   labels=['Low', 'Medium', 'High', 'Very High'])
    df_clean['income_tier'] = pd.cut(df_clean['annual_inc'],
                                      bins=[0, 40000, 80000, 150000, np.inf],
                                      labels=['Low', 'Mid', 'High', 'Very High'])
    result['default_rate_by_dti_tier'] = df_clean.groupby('dti_tier', observed=True)['default'].mean().to_dict()
    result['default_rate_by_income_tier'] = df_clean.groupby('income_tier', observed=True)['default'].mean().to_dict()

    # ---- 3. Categorical vs target (default rate by category) ----
    result['default_rate_by_grade'] = df_clean.groupby('grade', observed=True)['default'].mean().sort_index().to_dict()
    result['default_rate_by_purpose'] = df_clean.groupby('purpose', observed=True)['default'].mean().sort_values(ascending=False).to_dict()

    # ---- 4. Encoding (label-encode categoricals for modeling / feature importance) ----
    df_encoded = df_clean.copy()
    encoders = {}
    for col in CATEGORICAL_COLS + ['dti_tier', 'income_tier']:
        le = LabelEncoder()
        df_encoded[col + '_enc'] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    # ---- 5. Feature importance (quick RandomForest on encoded + numeric features) ----
    feature_cols = [c for c in NUMERIC_COLS] + [c + '_enc' for c in CATEGORICAL_COLS]
    X = df_encoded[feature_cols]
    y = df_encoded['default']
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    result['feature_importance'] = importances.to_dict()

    # ================= VISUALIZATIONS =================
    sns.set_style('whitegrid')

    # Univariate: default distribution
    plt.figure(figsize=(5, 4))
    df_clean['default'].value_counts().plot(kind='bar', color=['#4C72B0', '#C44E52'])
    plt.title('Class Distribution: Default vs Non-Default')
    plt.xlabel('Default (0=No, 1=Yes)'); plt.ylabel('Count')
    plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/01_class_distribution.png'); plt.close()

    # Univariate: int_rate distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df_clean['int_rate'], bins=30, kde=True, color='#55A868')
    plt.title('Distribution of Interest Rate')
    plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/02_int_rate_dist.png'); plt.close()

    # Univariate: annual_inc distribution (log scale, right-skewed)
    plt.figure(figsize=(6, 4))
    sns.histplot(df_clean['annual_inc'], bins=40, color='#8172B2')
    plt.title('Distribution of Annual Income')
    plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/03_annual_inc_dist.png'); plt.close()

    # Bivariate: default rate by grade
    plt.figure(figsize=(6, 4))
    pd.Series(result['default_rate_by_grade']).plot(kind='bar', color='#C44E52')
    plt.title('Default Rate by Loan Grade'); plt.ylabel('Default Rate')
    plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/04_default_rate_by_grade.png'); plt.close()

    # Bivariate: dti vs default (boxplot)
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df_clean, x='default', y='dti', hue='default',
                palette=['#4C72B0', '#C44E52'], legend=False)
    plt.title('Debt-to-Income Ratio by Default Status')
    plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/05_dti_by_default.png'); plt.close()

    # Bivariate: default rate by purpose
    plt.figure(figsize=(7, 4))
    pd.Series(result['default_rate_by_purpose']).plot(kind='barh', color='#DD8452')
    plt.title('Default Rate by Loan Purpose'); plt.xlabel('Default Rate')
    plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/06_default_rate_by_purpose.png'); plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap (Numeric Features + Default)')
    plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/07_correlation_heatmap.png'); plt.close()

    # Feature importance chart
    plt.figure(figsize=(7, 5))
    importances.plot(kind='barh', color='#4C72B0')
    plt.gca().invert_yaxis()
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout(); plt.savefig(f'{PLOTS_DIR}/08_feature_importance.png'); plt.close()

    result['plots_generated'] = 8
    return result


if __name__ == '__main__':
    df = load_data()
    pre = preprocess(df)
    eda_result = run_eda(pre['df_clean'])

    print("Correlation with target (top drivers):")
    for k, v in list(eda_result['correlation_with_target'].items())[:5]:
        print(f"  {k}: {v:.3f}")

    print("\nDefault rate by grade:", eda_result['default_rate_by_grade'])
    print("\nTop 5 feature importances:")
    for k, v in list(eda_result['feature_importance'].items())[:5]:
        print(f"  {k}: {v:.3f}")
    print(f"\n{eda_result['plots_generated']} plots saved to {PLOTS_DIR}")
