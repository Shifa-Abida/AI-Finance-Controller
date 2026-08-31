import pandas as pd


# ============================================================
# 1. MODEL RESULTS
# ============================================================

results = pd.DataFrame({
    "model": [
        "Baseline",
        "Linear Regression",
        "Random Forest",
        "Improved Random Forest"
    ],

    "mae_days": [
        2.64,
        2.61,
        2.61,
        2.54
    ]
})


# ============================================================
# 2. CALCULATE IMPROVEMENT FROM BASELINE
# ============================================================

baseline_mae = results.loc[
    results["model"] == "Baseline",
    "mae_days"
].iloc[0]


results["improvement_vs_baseline_days"] = (
    baseline_mae - results["mae_days"]
)


# ============================================================
# 3. CALCULATE IMPROVEMENT PERCENTAGE
# ============================================================

results["improvement_vs_baseline_percent"] = (
    results["improvement_vs_baseline_days"]
    / baseline_mae
    * 100
)


# ============================================================
# 4. FIND BEST MODEL
# ============================================================

best_model = results.loc[
    results["mae_days"].idxmin()
]


# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

print("\n========== FINAL MODEL COMPARISON ==========")

print(
    results.to_string(index=False)
)


print("\n========== BEST MODEL ==========")

print(
    "Model:",
    best_model["model"]
)

print(
    "MAE:",
    best_model["mae_days"],
    "days"
)

print(
    "Improvement vs baseline:",
    round(
        best_model["improvement_vs_baseline_days"],
        2
    ),
    "days"
)

print(
    "Improvement percentage:",
    round(
        best_model["improvement_vs_baseline_percent"],
        2
    ),
    "%"
)


# ============================================================
# 6. SAVE RESULTS
# ============================================================

results.to_csv(
    "results/final_model_comparison.csv",
    index=False
)


print("\nSaved:")
print(
    "results/final_model_comparison.csv"
)