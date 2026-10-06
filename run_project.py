import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import *
from src.woe_iv import WOEEncoder
from src.metrics import classification_metrics


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)


# ============================================================
# DATA CLEANING
# ============================================================

def clean_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False),
                errors="coerce"
            )
    return df


def prepare_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print("Loading modeling dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Keep only resolved loan outcomes.
    df = df[
        df[TARGET].isin(BAD_STATUSES | GOOD_STATUSES)
    ].copy()

    # Binary target:
    # 0 = Good / Fully Paid
    # 1 = Bad / Charged Off or Default
    df["target"] = (
        df[TARGET].isin(BAD_STATUSES).astype(int)
    )

    num_cols = [
        c for c in NUMERIC_FEATURES
        if c in df.columns
    ]

    cat_cols = [
        c for c in CATEGORICAL_FEATURES
        if c in df.columns
    ]

    df = clean_numeric(df, num_cols)

    features = num_cols + cat_cols

    df = df.dropna(
        subset=features,
        how="all"
    )

    print(f"Rows available for modeling: {len(df):,}")
    print(f"Good loans: {(df['target'] == 0).sum():,}")
    print(f"Bad loans: {(df['target'] == 1).sum():,}")

    return df, num_cols, cat_cols


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_data(df, features):

    X = df[features]
    y = df["target"]

    # First hold out 20% as untouched test data.
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Validation share relative to remaining train+validation data.
    validation_fraction = VALID_SIZE / (1 - TEST_SIZE)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=validation_fraction,
        stratify=y_train_valid,
        random_state=RANDOM_STATE
    )

    print("\nDataset split:")
    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_valid):,}")
    print(f"Test rows:       {len(X_test):,}")

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    )


# ============================================================
# TREE MODEL PREPROCESSING
# ============================================================

def tree_matrix(X_train, X_valid, X_test, cat_cols):

    train = X_train.copy()
    valid = X_valid.copy()
    test = X_test.copy()

    # Category mappings are learned from TRAINING DATA ONLY.
    for c in cat_cols:

        train_values = (
            train[c]
            .fillna("__MISSING__")
            .astype(str)
        )

        categories = {
            value: idx
            for idx, value in enumerate(
                train_values.unique()
            )
        }

        train[c] = train_values.map(categories)

        valid[c] = (
            valid[c]
            .fillna("__MISSING__")
            .astype(str)
            .map(categories)
            .fillna(-1)
        )

        test[c] = (
            test[c]
            .fillna("__MISSING__")
            .astype(str)
            .map(categories)
            .fillna(-1)
        )

    for c in train.columns:

        train[c] = pd.to_numeric(
            train[c],
            errors="coerce"
        )

        valid[c] = pd.to_numeric(
            valid[c],
            errors="coerce"
        )

        test[c] = pd.to_numeric(
            test[c],
            errors="coerce"
        )

    train = train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid = valid.replace(
        [np.inf, -np.inf],
        np.nan
    )

    test = test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Imputer is also fitted on TRAINING DATA ONLY.
    imputer = SimpleImputer(strategy="median")

    train = pd.DataFrame(
        imputer.fit_transform(train),
        columns=train.columns,
        index=train.index
    )

    valid = pd.DataFrame(
        imputer.transform(valid),
        columns=valid.columns,
        index=valid.index
    )

    test = pd.DataFrame(
        imputer.transform(test),
        columns=test.columns,
        index=test.index
    )

    return train, valid, test


# ============================================================
# PROFIT / APPROVAL POLICY
# ============================================================

def score_profit_data(scores, scored):

    scored = scored.copy()

    scored["PD"] = scores

    exposure = pd.to_numeric(
        scored[EXPOSURE_COL],
        errors="coerce"
    )

    rate = pd.to_numeric(
        scored[PRICING_COL],
        errors="coerce"
    ) / 100

    months = pd.to_numeric(
        scored[TERM_COL]
        .astype(str)
        .str.extract(r"(\d+)")[0],
        errors="coerce"
    ).fillna(36)

    interest_income = (
        exposure
        * rate
        * months
        / 12
    )

    scored["ExpectedLoss"] = (
        scored["PD"]
        * LGD
        * exposure
    )

    scored["ExpectedProfit"] = (
        interest_income
        - scored["ExpectedLoss"]
    )

    return scored


def build_profit_table(scores, scored):

    scored = score_profit_data(
        scores,
        scored
    )

    rows = []

    for cutoff in np.arange(
        0.01,
        0.501,
        0.01
    ):

        approved = scored[
            scored["PD"] <= cutoff
        ]

        if approved.empty:
            continue

        rows.append({

            "cutoff":
                cutoff,

            "approved_loans":
                len(approved),

            "approval_rate":
                len(approved) / len(scored),

            "predicted_bad_rate":
                approved["PD"].mean(),

            "actual_bad_rate":
                approved["target"].mean(),

            "expected_profit":
                approved["ExpectedProfit"].sum(),

            "profit_per_approved_loan":
                approved["ExpectedProfit"].mean()
        })

    return pd.DataFrame(rows)


def evaluate_fixed_cutoff(scores, scored, cutoff):

    scored = score_profit_data(
        scores,
        scored
    )

    approved = scored[
        scored["PD"] <= cutoff
    ].copy()

    if approved.empty:

        return {
            "cutoff": cutoff,
            "approved_loans": 0,
            "approval_rate": 0,
            "predicted_bad_rate": np.nan,
            "actual_bad_rate": np.nan,
            "expected_profit": 0,
            "profit_per_approved_loan": np.nan
        }

    return {

        "cutoff":
            cutoff,

        "approved_loans":
            len(approved),

        "approval_rate":
            len(approved) / len(scored),

        "predicted_bad_rate":
            approved["PD"].mean(),

        "actual_bad_rate":
            approved["target"].mean(),

        "expected_profit":
            approved["ExpectedProfit"].sum(),

        "profit_per_approved_loan":
            approved["ExpectedProfit"].mean()
    }


# ============================================================
# MAIN MODELING PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    df, num_cols, cat_cols = prepare_data()

    features = num_cols + cat_cols


    # --------------------------------------------------------
    # 2. SPLIT DATA
    # --------------------------------------------------------

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test
    ) = split_data(
        df,
        features
    )


    # --------------------------------------------------------
    # 3. WOE / IV
    # --------------------------------------------------------

    print("\nFitting WOE / IV transformation...")

    woe = WOEEncoder(n_bins=10)

    Xtr_woe = woe.fit_transform(
        X_train,
        y_train,
        cat_cols
    )

    Xv_woe = woe.transform(
        X_valid,
        cat_cols
    )

    Xt_woe = woe.transform(
        X_test,
        cat_cols
    )

    iv_table = woe.iv_table()

    iv_table.to_csv(
        OUTPUT_DIR / "feature_woe_iv.csv",
        index=False
    )

    selected = iv_table.loc[
        iv_table["IV"] >= 0.02,
        "feature"
    ].tolist()

    if not selected:

        selected = (
            iv_table
            .head(min(10, len(iv_table)))
            ["feature"]
            .tolist()
        )

    print(
        f"Selected WOE features: "
        f"{len(selected)}"
    )


    # --------------------------------------------------------
    # 4. LOGISTIC REGRESSION SCORECARD
    # --------------------------------------------------------

    print("\nTraining Logistic Regression scorecard...")

    smote = SMOTE(
        random_state=RANDOM_STATE
    )

    X_bal, y_bal = smote.fit_resample(
        Xtr_woe[selected],
        y_train
    )

    logit = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    logit.fit(
        X_bal,
        y_bal
    )

    # Validation probabilities are used for policy selection.
    p_valid_logit = logit.predict_proba(
        Xv_woe[selected]
    )[:, 1]

    # Test probabilities remain untouched until final evaluation.
    p_test_logit = logit.predict_proba(
        Xt_woe[selected]
    )[:, 1]


    # --------------------------------------------------------
    # 5. TREE PREPROCESSING
    # --------------------------------------------------------

    print("\nPreparing tree-model data...")

    (
        Xtr_tree,
        Xv_tree,
        Xt_tree
    ) = tree_matrix(
        X_train,
        X_valid,
        X_test,
        cat_cols
    )

    Xtr_tree_bal, ytr_tree_bal = SMOTE(
        random_state=RANDOM_STATE
    ).fit_resample(
        Xtr_tree,
        y_train
    )


    # --------------------------------------------------------
    # 6. XGBOOST
    # --------------------------------------------------------

    print("\nTraining XGBoost...")

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    xgb.fit(
        Xtr_tree_bal,
        ytr_tree_bal
    )

    p_test_xgb = xgb.predict_proba(
        Xt_tree
    )[:, 1]


    # --------------------------------------------------------
    # 7. LIGHTGBM
    # --------------------------------------------------------

    print("\nTraining LightGBM...")

    lgbm = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=-1
    )

    lgbm.fit(
        Xtr_tree_bal,
        ytr_tree_bal
    )

    p_test_lgbm = lgbm.predict_proba(
        Xt_tree
    )[:, 1]


    # --------------------------------------------------------
    # 8. FINAL TEST MODEL COMPARISON
    # --------------------------------------------------------

    print("\nCalculating final test metrics...")

    results = []

    models = [

        (
            "Logistic_WOE_Scorecard",
            p_test_logit
        ),

        (
            "XGBoost",
            p_test_xgb
        ),

        (
            "LightGBM",
            p_test_lgbm
        )
    ]

    for name, pred in models:

        row = classification_metrics(
            y_test,
            pred
        )

        row["model"] = name

        results.append(row)

    model_metrics = pd.DataFrame(
        results
    )[
        [
            "model",
            "AUC",
            "Gini",
            "KS"
        ]
    ].sort_values(
        "AUC",
        ascending=False
    )

    model_metrics.to_csv(
        OUTPUT_DIR / "model_metrics.csv",
        index=False
    )


    # ========================================================
    # 9. SELECT CUTOFF USING VALIDATION DATA
    # ========================================================

    print(
        "\nSelecting approval cutoff "
        "using VALIDATION data..."
    )

    validation_scored = df.loc[
        X_valid.index
    ].copy()

    validation_scored["target"] = y_valid

    validation_cutoff_table = build_profit_table(
        p_valid_logit,
        validation_scored
    )

    validation_cutoff_table.to_csv(
        OUTPUT_DIR /
        "validation_cutoff_analysis.csv",
        index=False
    )

    best_validation = (
        validation_cutoff_table.loc[
            validation_cutoff_table[
                "expected_profit"
            ].idxmax()
        ]
    )

    optimal_cutoff = float(
        best_validation["cutoff"]
    )


    # ========================================================
    # 10. APPLY FIXED CUTOFF TO UNTOUCHED TEST DATA
    # ========================================================

    print(
        "\nApplying validation-selected cutoff "
        "to TEST data..."
    )

    test_scored = df.loc[
        X_test.index
    ].copy()

    test_scored["target"] = y_test

    test_policy = evaluate_fixed_cutoff(
        p_test_logit,
        test_scored,
        optimal_cutoff
    )

    test_policy_df = pd.DataFrame(
        [test_policy]
    )

    test_policy_df.to_csv(
        OUTPUT_DIR /
        "test_policy_results.csv",
        index=False
    )


    # ========================================================
    # 11. WRITE SUMMARY
    # ========================================================

    with open(
        OUTPUT_DIR /
        "approval_cutoff_summary.txt",
        "w"
    ) as f:

        f.write(
            "Credit Risk Approval Policy\n"
        )

        f.write(
            "===========================\n\n"
        )

        f.write(
            "Cutoff selected using "
            "VALIDATION data.\n"
        )

        f.write(
            "Final policy evaluated using "
            "untouched TEST data.\n\n"
        )

        f.write(
            f"Optimal PD cutoff: "
            f"{optimal_cutoff:.2%}\n\n"
        )

        f.write(
            "FINAL TEST RESULTS\n"
        )

        f.write(
            "------------------\n"
        )

        f.write(
            f"Approval rate: "
            f"{test_policy['approval_rate']:.2%}\n"
        )

        f.write(
            f"Predicted bad rate: "
            f"{test_policy['predicted_bad_rate']:.2%}\n"
        )

        f.write(
            f"Actual bad rate: "
            f"{test_policy['actual_bad_rate']:.2%}\n"
        )

        f.write(
            f"Expected portfolio profit: "
            f"{test_policy['expected_profit']:,.2f}\n"
        )

        f.write(
            f"Expected profit per approved loan: "
            f"{test_policy['profit_per_approved_loan']:,.2f}\n"
        )


    # ========================================================
    # 12. CHARTS
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        validation_cutoff_table["cutoff"],
        validation_cutoff_table["approval_rate"],
        label="Approval rate"
    )

    ax.plot(
        validation_cutoff_table["cutoff"],
        validation_cutoff_table["actual_bad_rate"],
        label="Actual bad rate"
    )

    ax.axvline(
        optimal_cutoff,
        linestyle="--",
        label="Selected cutoff"
    )

    ax.set_xlabel(
        "PD cutoff"
    )

    ax.set_ylabel(
        "Rate"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "figures/approval_vs_bad_rate.png",
        dpi=150
    )

    plt.close(fig)


    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        validation_cutoff_table["cutoff"],
        validation_cutoff_table["expected_profit"]
    )

    ax.axvline(
        optimal_cutoff,
        linestyle="--"
    )

    ax.set_xlabel(
        "PD cutoff"
    )

    ax.set_ylabel(
        "Expected portfolio profit"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "figures/profit_vs_cutoff.png",
        dpi=150
    )

    plt.close(fig)


    # ========================================================
    # 13. TERMINAL RESULTS
    # ========================================================

    print(
        "\n========== MODEL COMPARISON =========="
    )

    print(
        model_metrics.to_string(
            index=False
        )
    )

    print(
        "\n========== VALIDATION POLICY =========="
    )

    print(
        best_validation.to_string()
    )

    print(
        "\n========== FINAL TEST POLICY =========="
    )

    print(
        pd.Series(
            test_policy
        ).to_string()
    )

    print(
        f"\nOutputs saved to: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()