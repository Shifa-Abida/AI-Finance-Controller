import pandas as pd


# ============================================================
# 1. LOAD AR INVOICES
# ============================================================

ar_df = pd.read_csv(
    "data/ar_invoices.csv"
)


# ============================================================
# 2. CONVERT DATES
# ============================================================

ar_df["invoice_date"] = pd.to_datetime(
    ar_df["invoice_date"]
)

ar_df["due_date"] = pd.to_datetime(
    ar_df["due_date"]
)

ar_df["actual_payment_date"] = pd.to_datetime(
    ar_df["actual_payment_date"]
)


# ============================================================
# 3. SORT BY CUSTOMER AND INVOICE DATE
# ============================================================

ar_df = ar_df.sort_values(
    ["customer_id", "invoice_date"]
).reset_index(drop=True)


# ============================================================
# 4. CREATE HISTORICAL CUSTOMER FEATURES
# ============================================================

ar_df["avg_delay"] = (
    ar_df.groupby("customer_id")["payment_delay_days"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

ar_df["median_delay"] = (
    ar_df.groupby("customer_id")["payment_delay_days"]
    .transform(lambda x: x.shift(1).expanding().median())
)

ar_df["std_delay"] = (
    ar_df.groupby("customer_id")["payment_delay_days"]
    .transform(lambda x: x.shift(1).expanding().std())
)

ar_df["invoice_count"] = (
    ar_df.groupby("customer_id")["payment_delay_days"]
    .transform(
        lambda x: x.shift(1).expanding().count()
    )
)


# ============================================================
# 5. CALCULATE OVERALL AVERAGE DELAY
# ============================================================

overall_avg_delay = ar_df[
    "payment_delay_days"
].mean()

print(
    "Overall average delay:",
    overall_avg_delay
)


# ============================================================
# 6. FILL MISSING CUSTOMER HISTORY
# ============================================================

ar_df["avg_delay"] = ar_df[
    "avg_delay"
].fillna(overall_avg_delay)

ar_df["median_delay"] = ar_df[
    "median_delay"
].fillna(overall_avg_delay)

ar_df["std_delay"] = ar_df[
    "std_delay"
].fillna(0)

ar_df["invoice_count"] = ar_df[
    "invoice_count"
].fillna(0)


# ============================================================
# 7. SELECT ML FEATURES
# ============================================================

ml_df = ar_df[
    [
        "invoice_id",
        "customer_id",
        "invoice_date",
        "amount",
        "avg_delay",
        "median_delay",
        "std_delay",
        "invoice_count",
        "payment_delay_days"
    ]
]


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n========== LEAKAGE-FREE ML FEATURES ==========")

print("Rows:", len(ml_df))

print("\nMissing values:")

print(
    ml_df.isnull().sum()
)

print("\nSample data:")

print(
    ml_df.head(10)
)


# ============================================================
# 9. SAVE DATASET
# ============================================================

ml_df.to_csv(
    "data/ml_features.csv",
    index=False
)

print("\nSaved:")
print("data/ml_features.csv")