import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "lending_club.csv"
OUTPUT_FILE = BASE_DIR / "data" / "lending_club_model.csv"


# --------------------------------------------------
# 2. COLUMNS NEEDED FOR OUR CREDIT RISK PROJECT
# --------------------------------------------------

columns_needed = [
    "loan_status",
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "application_type",
    "acc_open_past_24mths",
    "mort_acc",
    "pub_rec_bankruptcies",
    "tot_cur_bal",
    "total_rev_hi_lim",
]


# --------------------------------------------------
# 3. READ LARGE CSV IN CHUNKS
# --------------------------------------------------

chunks = []

print("Starting Lending Club data preparation...")
print("Input file:", INPUT_FILE)

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=columns_needed,
        chunksize=100000,
        low_memory=False
    ),
    start=1
):

    print(f"Processing chunk {chunk_number}...")

    # Keep only loans where the final outcome is known.
    chunk = chunk[
        chunk["loan_status"].isin(
            ["Fully Paid", "Charged Off", "Default"]
        )
    ].copy()

    chunks.append(chunk)


# --------------------------------------------------
# 4. COMBINE THE PROCESSED CHUNKS
# --------------------------------------------------

print("Combining processed chunks...")

df = pd.concat(chunks, ignore_index=True)


# --------------------------------------------------
# 5. CREATE BINARY DEFAULT TARGET
# --------------------------------------------------

df["default"] = df["loan_status"].map(
    {
        "Fully Paid": 0,
        "Charged Off": 1,
        "Default": 1
    }
)


# --------------------------------------------------
# 6. REMOVE DUPLICATES
# --------------------------------------------------

df = df.drop_duplicates()


# --------------------------------------------------
# 7. DISPLAY BASIC INFORMATION
# --------------------------------------------------

print()
print("Data preparation complete.")
print("--------------------------------")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print()
print("Loan status distribution:")
print(df["loan_status"].value_counts())

print()
print("Default target distribution:")
print(df["default"].value_counts())


# --------------------------------------------------
# 8. SAVE CLEAN MODELING DATASET
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Clean modeling dataset saved successfully:")
print(OUTPUT_FILE)