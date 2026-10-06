# Credit Default Risk Modeling & Profit-Optimized Approval Strategy

End-to-end credit risk project using Lending Club loan data to predict Probability of Default (PD), build an interpretable WOE/IV logistic scorecard, benchmark XGBoost and LightGBM, and optimize an approval cutoff for expected portfolio profitability.

## Workflow
1. Clean Lending Club origination data.
2. Define `Charged Off` / `Default` as bad and `Fully Paid` as good.
3. Split into train / validation / test using stratification.
4. Fit WOE bins and calculate Information Value (IV) on training data only.
5. Select variables using IV.
6. Apply SMOTE only to the training set.
7. Train logistic regression, XGBoost and LightGBM.
8. Evaluate AUC-ROC, Gini and KS.
9. Calculate expected loss and expected profit for candidate PD cutoffs.
10. Select the cutoff that maximizes expected portfolio profit.
11. Produce approval-rate vs bad-rate and profit-vs-cutoff charts.

## Dataset
Place a Lending Club CSV at:

`data/lending_club.csv`

The code expects a `loan_status` column and uses common Lending Club fields when available.

## Run
```bash
pip install -r requirements.txt
python run_project.py
```

Outputs are written to `outputs/`.

## Business logic
Expected Loss = PD × LGD × Exposure

Expected Profit = Interest Income − Expected Loss

The interest-income calculation is a simplified screening assumption. A production model should incorporate amortization, prepayment, recovery timing, funding cost, servicing cost, acquisition cost and capital cost.

## Key modeling controls
- No SMOTE before the train/test split.
- WOE/IV is learned only from training data.
- Test data remains untouched until final evaluation.
- Origination-time variables should be used; post-origination leakage should be excluded.
- Policy cutoff is optimized on an out-of-sample scoring set.
