import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# ============================================================
# 1. LOAD DATA
# ============================================================

ml_df = pd.read_csv(
    "data/ml_features.csv"
)

ar_df = pd.read_csv(
    "data/ar_invoices.csv"
)


# ============================================================
# 2. CONVERT DATES
# ============================================================

ml_df["invoice_date"] = pd.to_datetime(
    ml_df["invoice_date"]
)


# ============================================================
# 3. SORT BY DATE
# ============================================================

ml_df = ml_df.sort_values(
    "invoice_date"
).reset_index(drop=True)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(ml_df) * 0.8)

train_df = ml_df.iloc[:split_index]
test_df = ml_df.iloc[split_index:]


# ============================================================
# 5. FEATURES
# ============================================================

features = [
    "amount",
    "avg_delay",
    "median_delay",
    "std_delay",
    "invoice_count"
]


X_train = train_df[features]
X_test = test_df[features]

y_train = train_df["payment_delay_days"]
y_test = test_df["payment_delay_days"]


# ============================================================
# 6. TRAIN LINEAR REGRESSION
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ============================================================
# 7. ML PREDICTIONS
# ============================================================

ml_predictions = model.predict(
    X_test
)


# ============================================================
# 8. BASELINE PREDICTIONS
# ============================================================

baseline_predictions = test_df[
    "avg_delay"
].values


# ============================================================
# 9. CALCULATE MAE
# ============================================================

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

ml_mae = mean_absolute_error(
    y_test,
    ml_predictions
)


# ============================================================
# 10. CALCULATE DIFFERENCE
# ============================================================

difference = ml_mae - baseline_mae


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n========== MODEL COMPARISON ==========")

print(
    "Test invoices:",
    len(test_df)
)

print(
    "\nBaseline MAE:",
    round(baseline_mae, 2),
    "days"
)

print(
    "Linear Regression MAE:",
    round(ml_mae, 2),
    "days"
)

print(
    "Difference:",
    round(difference, 2),
    "days"
)


# ============================================================
# 12. DETERMINE WINNER
# ============================================================

if ml_mae < baseline_mae:

    print("\nWinner: Linear Regression")

else:

    print("\nWinner: Baseline")


# ============================================================
# 13. SAMPLE COMPARISON
# ============================================================

comparison = test_df[
    [
        "invoice_id",
        "customer_id",
        "payment_delay_days"
    ]
].copy()

comparison["baseline_prediction"] = (
    baseline_predictions
)

comparison["ml_prediction"] = (
    ml_predictions
)

comparison["baseline_error"] = (
    abs(
        comparison["payment_delay_days"]
        - comparison["baseline_prediction"]
    )
)

comparison["ml_error"] = (
    abs(
        comparison["payment_delay_days"]
        - comparison["ml_prediction"]
    )
)


print("\n========== SAMPLE COMPARISON ==========")

print(
    comparison.head(10)
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "results/model_comparison.csv"
)