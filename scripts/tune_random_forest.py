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
# 5. FEATURES
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

y_train = train_df["payment_delay_days"]
y_test = test_df["payment_delay_days"]


# ============================================================
# 6. HYPERPARAMETER COMBINATIONS
# ============================================================

models = [

    {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    },

    {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    }
]


# ============================================================
# 7. TEST EACH MODEL
# ============================================================

results = []


print("\n========== RANDOM FOREST TUNING ==========")


for i, params in enumerate(models, start=1):

    model = RandomForestRegressor(
        **params,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    results.append({
        "model_number": i,
        **params,
        "mae_days": mae
    })

    print(
        f"Model {i} | "
        f"Trees: {params['n_estimators']} | "
        f"Depth: {params['max_depth']} | "
        f"Split: {params['min_samples_split']} | "
        f"Leaf: {params['min_samples_leaf']} | "
        f"MAE: {mae:.2f}"
    )


# ============================================================
# 8. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 9. SORT BY MAE
# ============================================================

results_df = results_df.sort_values(
    "mae_days"
).reset_index(drop=True)


# ============================================================
# 10. FIND BEST MODEL
# ============================================================

best_model = results_df.iloc[0]


print("\n========== TUNING RESULTS ==========")

print(
    results_df.to_string(index=False)
)


print("\n========== BEST CONFIGURATION ==========")

print(
    "Trees:",
    best_model["n_estimators"]
)

print(
    "Max depth:",
    best_model["max_depth"]
)

print(
    "Min samples split:",
    best_model["min_samples_split"]
)

print(
    "Min samples leaf:",
    best_model["min_samples_leaf"]
)

print(
    "MAE:",
    round(best_model["mae_days"], 2),
    "days"
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "results/random_forest_tuning.csv",
    index=False
)


print("\nSaved:")
print(
    "results/random_forest_tuning.csv"
)