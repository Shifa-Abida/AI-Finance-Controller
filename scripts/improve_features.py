import pandas as pd


# ============================================================
# 1. LOAD AR DATA
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

ar_df["actual_payment_date"] = pd.to_datetime(
    ar_df["actual_payment_date"]
)


# ============================================================
# 3. SORT BY CUSTOMER AND DATE
# ============================================================

ar_df = ar_df.sort_values(
    ["customer_id", "invoice_date"]
).reset_index(drop=True)


# ============================================================
# 4. CREATE PREVIOUS CUSTOMER HISTORY
# ============================================================

ar_df["previous_invoice_count"] = (
    ar_df.groupby("customer_id")
    .cumcount()
)


# ============================================================
# 5. PREVIOUS AVERAGE DELAY
# ============================================================

ar_df["previous_avg_delay"] = (
    ar_df.groupby("customer_id")[
        "payment_delay_days"
    ]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)


# ============================================================
# 6. PREVIOUS MEDIAN DELAY
# ============================================================

ar_df["previous_median_delay"] = (
    ar_df.groupby("customer_id")[
        "payment_delay_days"
    ]
    .transform(
        lambda x: x.shift(1).expanding().median()
    )
)


# ============================================================
# 7. PREVIOUS DELAY STANDARD DEVIATION
# ============================================================

ar_df["previous_std_delay"] = (
    ar_df.groupby("customer_id")[
        "payment_delay_days"
    ]
    .transform(
        lambda x: x.shift(1).expanding().std()
    )
)


# ============================================================
# 8. PREVIOUS ON-TIME RATIO
# ============================================================

ar_df["was_on_time"] = (
    ar_df["payment_delay_days"] <= 0
).astype(int)


ar_df["previous_on_time_ratio"] = (
    ar_df.groupby("customer_id")[
        "was_on_time"
    ]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)


# ============================================================
# 9. PREVIOUS LATE RATIO
# ============================================================

ar_df["previous_late_ratio"] = (
    1 - ar_df["previous_on_time_ratio"]
)


# ============================================================
# 10. DAYS SINCE PREVIOUS INVOICE
# ============================================================

ar_df["previous_invoice_date"] = (
    ar_df.groupby("customer_id")[
        "invoice_date"
    ].shift(1)
)


ar_df["days_since_previous_invoice"] = (
    ar_df["invoice_date"]
    - ar_df["previous_invoice_date"]
).dt.days


# ============================================================
# 11. REMOVE HELPER COLUMNS
# ============================================================

ar_df = ar_df.drop(
    columns=[
        "was_on_time",
        "previous_invoice_date"
    ]
)


# ============================================================
# 12. HANDLE FIRST-INVOICE CASE
# ============================================================

overall_avg_delay = ar_df[
    "payment_delay_days"
].mean()


ar_df["previous_avg_delay"] = (
    ar_df["previous_avg_delay"]
    .fillna(overall_avg_delay)
)


ar_df["previous_median_delay"] = (
    ar_df["previous_median_delay"]
    .fillna(overall_avg_delay)
)


ar_df["previous_std_delay"] = (
    ar_df["previous_std_delay"]
    .fillna(0)
)


ar_df["previous_on_time_ratio"] = (
    ar_df["previous_on_time_ratio"]
    .fillna(0)
)


ar_df["previous_late_ratio"] = (
    ar_df["previous_late_ratio"]
    .fillna(1)
)


ar_df["days_since_previous_invoice"] = (
    ar_df["days_since_previous_invoice"]
    .fillna(0)
)


# ============================================================
# 13. SELECT ML FEATURES
# ============================================================

ml_features = [
    "invoice_id",
    "customer_id",
    "invoice_date",
    "amount",

    "previous_avg_delay",
    "previous_median_delay",
    "previous_std_delay",
    "previous_on_time_ratio",
    "previous_late_ratio",
    "days_since_previous_invoice",

    "payment_delay_days"
]


improved_df = ar_df[
    ml_features
].copy()


# ============================================================
# 14. SORT BY DATE
# ============================================================

improved_df = improved_df.sort_values(
    "invoice_date"
).reset_index(drop=True)


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

print("\n========== IMPROVED ML FEATURES ==========")

print(
    "Rows:",
    len(improved_df)
)

print("\nMissing values:")

print(
    improved_df.isnull().sum()
)

print("\nSample data:")

print(
    improved_df.head(10)
)


# ============================================================
# 16. SAVE DATASET
# ============================================================

improved_df.to_csv(
    "data/improved_ml_features.csv",
    index=False
)

print("\nSaved:")

print(
    "data/improved_ml_features.csv"
)