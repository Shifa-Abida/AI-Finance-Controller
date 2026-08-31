import pandas as pd

from sklearn.linear_model import LinearRegression


# ============================================================
# 1. LOAD LEAKAGE-FREE ML DATA
# ============================================================

ml_df = pd.read_csv(
    "data/ml_features.csv"
)


# ============================================================
# 2. SORT BY DATE
# ============================================================

ml_df["invoice_date"] = pd.to_datetime(
    ml_df["invoice_date"]
)

ml_df = ml_df.sort_values(
    "invoice_date"
).reset_index(drop=True)


# ============================================================
# 3. CREATE TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(ml_df) * 0.8)

train_df = ml_df.iloc[:split_index]
test_df = ml_df.iloc[split_index:]


# ============================================================
# 4. DEFINE FEATURES (X)
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
# 5. DEFINE TARGET (y)
# ============================================================

y_train = train_df[
    "payment_delay_days"
]

y_test = test_df[
    "payment_delay_days"
]


# ============================================================
# 6. CREATE MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 7. TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n========== LINEAR REGRESSION ==========")

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nSample predictions:")

for i in range(10):

    print(
        "Actual:",
        y_test.iloc[i],
        "| Predicted:",
        round(predictions[i], 2)
    )