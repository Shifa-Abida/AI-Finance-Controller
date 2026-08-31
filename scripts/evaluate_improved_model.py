import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# 1. LOAD IMPROVED ML DATA
# ============================================================

ml_df = pd.read_csv(
    "data/improved_ml_features.csv"
)


# ============================================================
# 2. CONVERT DATE
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
# 5. DEFINE IMPROVED FEATURES
# ============================================================

features = [
    "amount",
    "previous_avg_delay",
    "previous_median_delay",
    "previous_std_delay",
    "previous_on_time_ratio",
    "previous_late_ratio",
    "days_since_previous_invoice"
]


X_train = train_df[features]
X_test = test_df[features]


# ============================================================
# 6. DEFINE TARGET
# ============================================================

y_train = train_df[
    "payment_delay_days"
]

y_test = test_df[
    "payment_delay_days"
]


# ============================================================
# 7. CREATE RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# 10. CALCULATE MAE
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n========== IMPROVED MODEL EVALUATION ==========")

print(
    "Number of invoices evaluated:",
    len(y_test)
)

print(
    "Improved Random Forest MAE:",
    round(mae, 2),
    "days"
)


# ============================================================
# 12. COMPARE WITH OLD RANDOM FOREST
# ============================================================

old_mae = 2.61

difference = mae - old_mae

print(
    "\nOld Random Forest MAE:",
    old_mae,
    "days"
)

print(
    "Improved Random Forest MAE:",
    round(mae, 2),
    "days"
)

print(
    "Difference:",
    round(difference, 2),
    "days"
)


# ============================================================
# 13. DETERMINE RESULT
# ============================================================

if mae < old_mae:

    print(
        "\nResult: Improved features improved the model."
    )

elif mae > old_mae:

    print(
        "\nResult: Improved features did not improve the model."
    )

else:

    print(
        "\nResult: Both models performed equally."
    )