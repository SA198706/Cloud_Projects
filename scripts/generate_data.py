"""
generate_data.py
Generates a synthetic Lending-Club-style loan dataset (~10,000 rows)
and saves it to ../data/loan_data_raw.csv

Run from the scripts/ directory:
    python generate_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 10_000

rng = np.random.default_rng(SEED)

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "loan_data_raw.csv"

GRADES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
GRADE_PROBS = [0.20, 0.25, 0.22, 0.15, 0.10, 0.05, 0.03]

PURPOSE_OPTS = [
    'debt_consolidation', 'credit_card', 'home_improvement', 'other',
    'major_purchase', 'medical', 'small_business', 'car', 'vacation',
    'moving', 'wedding', 'house', 'educational', 'renewable_energy',
]
HOME_OPTS = ['MORTGAGE', 'RENT', 'OWN', 'OTHER', 'NONE']
VERIFY_OPTS = ['Verified', 'Source Verified', 'Not Verified']
EMP_LENGTHS = ['< 1 year', '1 year', '2 years', '3 years', '4 years',
               '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']


def _sub_grade(grade):
    n = rng.integers(1, 6)
    return f"{grade}{n}"


def build_dataset(n=N):
    grades = rng.choice(GRADES, size=n, p=GRADE_PROBS)
    grade_idx = np.array([GRADES.index(g) for g in grades])  # 0=A .. 6=G

    # Interest rate rises with grade risk
    int_rate = 5.5 + grade_idx * 3.5 + rng.normal(0, 1.5, n)
    int_rate = np.clip(int_rate, 4.0, 30.99).round(2)

    loan_amnt = rng.choice(
        [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
        size=n,
        p=[0.12, 0.20, 0.18, 0.17, 0.12, 0.10, 0.06, 0.05],
    ).astype(float)

    term = rng.choice(['36 months', '60 months'], size=n, p=[0.70, 0.30])
    term_months = np.where(term == '36 months', 36, 60).astype(float)
    installment = (loan_amnt * (int_rate / 100 / 12) /
                   (1 - (1 + int_rate / 100 / 12) ** (-term_months))).round(2)

    annual_inc = rng.lognormal(mean=10.9, sigma=0.7, size=n).round(2)
    annual_inc = np.clip(annual_inc, 15_000, 500_000)

    dti = rng.gamma(shape=2.5, scale=6, size=n).round(2)
    dti = np.clip(dti, 0, 60)

    delinq_2yrs = rng.choice([0, 1, 2, 3], size=n, p=[0.82, 0.12, 0.04, 0.02])
    inq_last_6mths = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.45, 0.30, 0.15, 0.07, 0.03])
    open_acc = rng.integers(1, 30, size=n).astype(float)
    pub_rec = rng.choice([0, 1, 2], size=n, p=[0.88, 0.10, 0.02])
    revol_bal = rng.lognormal(mean=8.5, sigma=1.1, size=n).round(2)
    revol_util = rng.beta(2, 3, size=n) * 100
    revol_util = revol_util.round(1)
    total_acc = (open_acc + rng.integers(0, 20, size=n)).astype(float)

    emp_length = rng.choice(EMP_LENGTHS, size=n)
    home_ownership = rng.choice(HOME_OPTS, size=n, p=[0.44, 0.38, 0.12, 0.04, 0.02])
    verification_status = rng.choice(VERIFY_OPTS, size=n, p=[0.38, 0.32, 0.30])
    purpose = rng.choice(PURPOSE_OPTS, size=n,
                          p=[0.35, 0.18, 0.10, 0.10, 0.05, 0.04,
                             0.04, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01])
    sub_grade = np.array([_sub_grade(g) for g in grades])

    # Default probability rises with grade, dti, int_rate
    log_odds = (
        -3.5
        + 0.35 * grade_idx
        + 0.04 * dti
        + 0.06 * (int_rate - 10)
        + 0.3 * delinq_2yrs
        + 0.15 * inq_last_6mths
        + rng.logistic(0, 1, n) * 0.3
    )
    default_prob = 1 / (1 + np.exp(-log_odds))
    is_default = rng.random(n) < default_prob

    loan_status = np.where(is_default, 'Charged Off', 'Fully Paid')

    # Inject ~4% missingness into selected numeric columns
    for col_arr, rate in [(annual_inc, 0.04), (dti, 0.035),
                           (revol_util, 0.03)]:
        mask = rng.random(n) < rate
        col_arr[mask] = np.nan

    df = pd.DataFrame({
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': int_rate,
        'installment': installment,
        'grade': grades,
        'sub_grade': sub_grade,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'annual_inc': annual_inc,
        'verification_status': verification_status,
        'loan_status': loan_status,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs.astype(float),
        'inq_last_6mths': inq_last_6mths.astype(float),
        'open_acc': open_acc,
        'pub_rec': pub_rec.astype(float),
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'purpose': purpose,
    })

    return df


if __name__ == '__main__':
    df = build_dataset()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    default_rate = (df['loan_status'] == 'Charged Off').mean()
    print(f"Generated {len(df)} rows  |  default rate = {default_rate:.3f}")
    print(f"Saved to {OUT_PATH}")
