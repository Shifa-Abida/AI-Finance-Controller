import pandas as pd


# ============================================================
# 1. LOAD ML DATASET
# ============================================================

ml_df = pd.read_csv(
    "data/ml_dataset.csv"
)


# ============================================================
# 2. LOAD ORIGINAL AR DATA
# ============================================================

ar_df = pd.read_csv(
    "data/ar_invoices.csv"
)


# ============================================================
# 3. ADD INVOICE DATE
# ============================================================

ml_df["invoice_date"] = pd.to_datetime(
    ar_df["invoice_date"]
)


# ============================================================
# 4. SORT BY DATE
# ============================================================

ml_df = ml_df.sort_values(
    "invoice_date"
).reset_index(drop=True)


# ============================================================
# 5. CREATE TIME-BASED SPLIT
# ============================================================

split_index = int(len(ml_df) * 0.8)

train_df = ml_df.iloc[:split_index]
test_df = ml_df.iloc[split_index:]


# ============================================================
# 6. DISPLAY INFORMATION
# ============================================================

print("\n========== TRAIN / TEST SPLIT ==========")

print("Total rows:", len(ml_df))

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))

print("\nTraining period:")
print(
    train_df["invoice_date"].min(),
    "to",
    train_df["invoice_date"].max()
)

print("\nTesting period:")
print(
    test_df["invoice_date"].min(),
    "to",
    test_df["invoice_date"].max()
)


# ============================================================
# 7. SAVE DATASETS
# ============================================================

train_df.to_csv(
    "data/train.csv",
    index=False
)

test_df.to_csv(
    "data/test.csv",
    index=False
)

print("\nSaved:")
print("data/train.csv")
print("data/test.csv")