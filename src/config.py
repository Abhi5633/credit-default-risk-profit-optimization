from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "lending_club_model.csv"
OUTPUT_DIR = ROOT / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALID_SIZE = 0.20

TARGET = "loan_status"
BAD_STATUSES = {"Charged Off", "Default"}
GOOD_STATUSES = {"Fully Paid"}

NUMERIC_FEATURES = [
    "loan_amnt", "installment", "annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "fico_range_low",
    "fico_range_high", "acc_open_past_24mths", "mort_acc",
    "pub_rec_bankruptcies", "tot_cur_bal", "total_rev_hi_lim"
]

CATEGORICAL_FEATURES = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "initial_list_status",
    "application_type"
]

PRICING_COL = "int_rate"
EXPOSURE_COL = "funded_amnt"
TERM_COL = "term"
LGD = 0.45
