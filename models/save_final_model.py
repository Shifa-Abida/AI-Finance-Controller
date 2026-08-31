import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor


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
# 5. DEFINE FEATURES
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

y_train = train_df[
    "payment_delay_days"
]


# ============================================================
# 6. CREATE FINAL MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# ============================================================
# 7. TRAIN FINAL MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# 8. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/payment_delay_model.pkl"
)


# ============================================================
# 9. SAVE FEATURE LIST
# ============================================================

with open(
    "models/model_features.txt",
    "w"
) as f:

    for feature in features:
        f.write(feature + "\n")


# ============================================================
# 10. DISPLAY RESULT
# ============================================================

print("\n========== FINAL MODEL ==========")

print(
    "Model: Random Forest"
)

print(
    "Number of trees: 200"
)

print(
    "Training rows:",
    len(X_train)
)

print(
    "Features:",
    len(features)
)

print("\nModel saved:")
print(
    "models/payment_delay_model.pkl"
)

print("\nFeature list saved:")
print(
    "models/model_features.txt"
)