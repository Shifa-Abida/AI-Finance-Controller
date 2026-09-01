import pandas as pd


# ============================================================
# 1. LOAD DAILY CASH FORECAST
# ============================================================

daily_df = pd.read_csv(
    "results/daily_cash_forecast.csv"
)

daily_df["date"] = pd.to_datetime(
    daily_df["date"]
)


# ============================================================
# 2. LOAD UNCERTAINTY BANDS
# ============================================================

uncertainty_df = pd.read_csv(
    "results/cash_uncertainty_bands.csv"
)

uncertainty_df["date"] = pd.to_datetime(
    uncertainty_df["date"]
)


# ============================================================
# 3. LOAD CASH SHORTAGE ANALYSIS
# ============================================================

risk_df = pd.read_csv(
    "results/cash_shortage_analysis.csv"
)

risk_df["date"] = pd.to_datetime(
    risk_df["date"]
)


# ============================================================
# 4. LOAD FORECAST HORIZONS
# ============================================================

horizon_df = pd.read_csv(
    "results/cash_forecast_horizons.csv"
)

horizon_df["forecast_date"] = pd.to_datetime(
    horizon_df["forecast_date"]
)


# ============================================================
# 5. SELECT DAILY FORECAST COLUMNS
# ============================================================

daily_df = daily_df[
    [
        "date",
        "expected_inflow",
        "expected_outflow",
        "net_cash_flow",
        "projected_cash_balance"
    ]
]


# ============================================================
# 6. SELECT UNCERTAINTY COLUMNS
# ============================================================

uncertainty_df = uncertainty_df[
    [
        "date",
        "lower_cash",
        "base_cash",
        "upper_cash"
    ]
]


# ============================================================
# 7. SELECT RISK COLUMNS
# ============================================================

risk_df = risk_df[
    [
        "date",
        "cash_buffer",
        "risk_status"
    ]
]


# ============================================================
# 8. MERGE DAILY FORECAST + UNCERTAINTY
# ============================================================

final_df = daily_df.merge(
    uncertainty_df,
    on="date",
    how="outer"
)


# ============================================================
# 9. MERGE RISK INFORMATION
# ============================================================

final_df = final_df.merge(
    risk_df,
    on="date",
    how="outer"
)


# ============================================================
# 10. SORT BY DATE
# ============================================================

final_df = final_df.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# 11. ADD FORECAST HORIZONS
# ============================================================

horizon_lookup = horizon_df.rename(
    columns={
        "forecast_date": "date"
    }
)

horizon_lookup = horizon_lookup[
    [
        "date",
        "cash_after_30_days",
        "cash_after_60_days",
        "cash_after_90_days"
    ]
]


final_df = final_df.merge(
    horizon_lookup,
    on="date",
    how="left"
)


# ============================================================
# 12. FORMAT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "expected_inflow",
    "expected_outflow",
    "net_cash_flow",
    "projected_cash_balance",
    "lower_cash",
    "base_cash",
    "upper_cash",
    "cash_buffer",
    "cash_after_30_days",
    "cash_after_60_days",
    "cash_after_90_days"
]

for column in numeric_columns:

    if column in final_df.columns:

        final_df[column] = (
            final_df[column]
            .round(2)
        )


# ============================================================
# 13. DISPLAY FINAL RESULTS
# ============================================================

print("\n========== FINAL CASH FORECAST ==========")

print(
    "Total forecast dates:",
    len(final_df)
)

print(
    "Warning dates:",
    (final_df["risk_status"] == "WARNING").sum()
)

print(
    "Critical dates:",
    (final_df["risk_status"] == "CRITICAL").sum()
)

print(
    "Safe dates:",
    (final_df["risk_status"] == "SAFE").sum()
)


print("\nSample final forecast:")

print(
    final_df.head(15)
)


# ============================================================
# 14. FIND WORST CASH POSITION
# ============================================================

worst_row = final_df.loc[
    final_df["lower_cash"].idxmin()
]


print("\n========== WORST-CASE CASH POSITION ==========")

print(
    "Date:",
    worst_row["date"].date()
)

print(
    "Lower cash: ₹",
    worst_row["lower_cash"]
)

print(
    "Base cash: ₹",
    worst_row["base_cash"]
)

print(
    "Upper cash: ₹",
    worst_row["upper_cash"]
)

print(
    "Risk:",
    worst_row["risk_status"]
)


# ============================================================
# 15. SAVE MASTER FORECAST
# ============================================================

final_df.to_csv(
    "results/final_cash_forecast.csv",
    index=False
)


print("\nSaved:")

print(
    "results/final_cash_forecast.csv"
)