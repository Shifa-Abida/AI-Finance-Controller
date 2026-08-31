import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

ar_df = pd.read_csv(
    "data/ar_invoices.csv"
)

customer_behavior = pd.read_csv(
    "data/customer_behavior.csv"
)


# ============================================================
# 2. SELECT REQUIRED INVOICE COLUMNS
# ============================================================

invoice_features = ar_df[
    [
        "invoice_id",
        "customer_id",
        "amount",
        "payment_delay_days"
    ]
]


# ============================================================
# 3. MERGE CUSTOMER BEHAVIOR
# ============================================================

ml_df = invoice_features.merge(
    customer_behavior,
    on="customer_id",
    how="left"
)


# ============================================================
# 4. SELECT FINAL ML COLUMNS
# ============================================================

ml_df = ml_df[
    [
        "invoice_id",
        "customer_id",
        "amount",
        "avg_delay",
        "median_delay",
        "std_delay",
        "invoice_count",
        "payment_delay_days"
    ]
]


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n========== ML DATASET ==========")

print("Rows:", len(ml_df))

print("\nMissing values:")
print(ml_df.isnull().sum())


# ============================================================
# 6. DISPLAY SAMPLE
# ============================================================

print("\nSample ML data:")

print(
    ml_df.head(10)
)


# ============================================================
# 7. SAVE ML DATASET
# ============================================================

ml_df.to_csv(
    "data/ml_dataset.csv",
    index=False
)

print("\nML dataset saved to:")
print("data/ml_dataset.csv")