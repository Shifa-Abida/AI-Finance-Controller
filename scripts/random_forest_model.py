import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# 1. LOAD LEAKAGE-FREE ML DATA
# ============================================================

ml_df = pd.read_csv(
    "data/ml_features.csv"
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
# 5. DEFINE FEATURES
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
# 7. CREATE RANDOM FOREST MODEL
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

print("\n========== RANDOM FOREST ==========")

print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)

print(
    "\nRandom Forest MAE:",
    round(mae, 2),
    "days"
)


# ============================================================
# 12. SAMPLE PREDICTIONS
# ============================================================

print("\n========== SAMPLE PREDICTIONS ==========")

for i in range(10):

    print(
        "Actual:",
        y_test.iloc[i],
        "| Predicted:",
        round(predictions[i], 2)
    )