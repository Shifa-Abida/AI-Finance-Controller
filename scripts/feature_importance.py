import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. LOAD ML DATA
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

y_train = train_df[
    "payment_delay_days"
]


# ============================================================
# 6. CREATE RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# 8. GET FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_


# ============================================================
# 9. CREATE RESULTS TABLE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": importance
})


# ============================================================
# 10. SORT BY IMPORTANCE
# ============================================================

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n========== FEATURE IMPORTANCE ==========")

for _, row in feature_importance.iterrows():

    print(
        row["feature"],
        "| Importance:",
        round(row["importance"], 4)
    )


# ============================================================
# 12. SAVE RESULTS
# ============================================================

feature_importance.to_csv(
    "results/feature_importance.csv",
    index=False
)

print("\nSaved:")
print("results/feature_importance.csv")