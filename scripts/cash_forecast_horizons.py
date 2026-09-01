import pandas as pd


# ============================================================
# 1. LOAD DAILY CASH FORECAST
# ============================================================

forecast_df = pd.read_csv(
    "results/daily_cash_forecast.csv"
)

forecast_df["date"] = pd.to_datetime(
    forecast_df["date"]
)


# ============================================================
# 2. SORT BY DATE
# ============================================================

forecast_df = forecast_df.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# 3. CREATE 30 / 60 / 90 DAY HORIZONS
# ============================================================

results = []


for _, row in forecast_df.iterrows():

    forecast_date = row["date"]

    current_cash = row[
        "projected_cash_balance"
    ]


    # --------------------------------------------------------
    # Target dates
    # --------------------------------------------------------

    date_30 = (
        forecast_date
        + pd.Timedelta(days=30)
    )

    date_60 = (
        forecast_date
        + pd.Timedelta(days=60)
    )

    date_90 = (
        forecast_date
        + pd.Timedelta(days=90)
    )


    # --------------------------------------------------------
    # Find projected cash at target dates
    # --------------------------------------------------------

    cash_30 = forecast_df.loc[
        forecast_df["date"] == date_30,
        "projected_cash_balance"
    ]

    cash_60 = forecast_df.loc[
        forecast_df["date"] == date_60,
        "projected_cash_balance"
    ]

    cash_90 = forecast_df.loc[
        forecast_df["date"] == date_90,
        "projected_cash_balance"
    ]


    # --------------------------------------------------------
    # Convert to values
    # --------------------------------------------------------

    cash_30 = (
        cash_30.iloc[0]
        if not cash_30.empty
        else None
    )

    cash_60 = (
        cash_60.iloc[0]
        if not cash_60.empty
        else None
    )

    cash_90 = (
        cash_90.iloc[0]
        if not cash_90.empty
        else None
    )


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({

        "forecast_date":
            forecast_date.date(),

        "current_projected_cash":
            round(current_cash, 2),

        "cash_after_30_days":
            None if cash_30 is None
            else round(cash_30, 2),

        "cash_after_60_days":
            None if cash_60 is None
            else round(cash_60, 2),

        "cash_after_90_days":
            None if cash_90 is None
            else round(cash_90, 2)

    })


# ============================================================
# 4. CREATE DATAFRAME
# ============================================================

horizon_df = pd.DataFrame(
    results
)


# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

print("\n========== CASH FORECAST HORIZONS ==========")

print(
    "Number of forecast dates:",
    len(horizon_df)
)

print("\nSample horizons:")

print(
    horizon_df.head(10)
)


# ============================================================
# 6. CHECK AVAILABLE HORIZONS
# ============================================================

print("\nAvailable 30-day forecasts:",
      horizon_df["cash_after_30_days"].notna().sum())

print("Available 60-day forecasts:",
      horizon_df["cash_after_60_days"].notna().sum())

print("Available 90-day forecasts:",
      horizon_df["cash_after_90_days"].notna().sum())


# ============================================================
# 7. SAVE RESULTS
# ============================================================

horizon_df.to_csv(
    "results/cash_forecast_horizons.csv",
    index=False
)


print("\nSaved:")
print(
    "results/cash_forecast_horizons.csv"
)
