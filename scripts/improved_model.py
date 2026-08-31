import pandas as pd

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
# 10. DISPLAY SAMPLE PREDICTIONS
# ============================================================

print("\n========== IMPROVED RANDOM FOREST ==========")

print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)

print("\nSample predictions:")

for i in range(10):

    print(
        "Actual:",
        y_test.iloc[i],
        "| Predicted:",
        round(predictions[i], 2)
    )


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": importance
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


print("\n========== FEATURE IMPORTANCE ==========")

for _, row in feature_importance.iterrows():

    print(
        row["feature"],
        "| Importance:",
        round(row["importance"], 4)
    )


# ============================================================
# 12. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "results/improved_feature_importance.csv",
    index=False
)


print("\nSaved:")
print("results/improved_feature_importance.csv")