import pandas as pd


# ============================================================
# 1. LOAD EXPECTED AR INFLOWS
# ============================================================

ar_df = pd.read_csv(
    "results/expected_ar_inflows.csv"
)

ar_df["predicted_payment_date"] = pd.to_datetime(
    ar_df["predicted_payment_date"]
)


# ============================================================
# 2. LOAD EXPECTED AP OUTFLOWS
# ============================================================

ap_df = pd.read_csv(
    "results/expected_ap_outflows.csv"
)

ap_df["expected_payment_date"] = pd.to_datetime(
    ap_df["expected_payment_date"]
)


# ============================================================
# 3. LOAD HISTORICAL CASH BALANCE
# ============================================================

cash_df = pd.read_csv(
    "data/cash_balance.csv"
)

cash_df["date"] = pd.to_datetime(
    cash_df["date"]
)


# ============================================================
# 4. AGGREGATE AR INFLOWS BY DATE
# ============================================================

daily_ar = (
    ar_df
    .groupby("predicted_payment_date")["expected_inflow"]
    .sum()
    .reset_index()
)

daily_ar = daily_ar.rename(
    columns={
        "predicted_payment_date": "date"
    }
)


# ============================================================
# 5. AGGREGATE AP OUTFLOWS BY DATE
# ============================================================

daily_ap = (
    ap_df
    .groupby("expected_payment_date")["expected_outflow"]
    .sum()
    .reset_index()
)

daily_ap = daily_ap.rename(
    columns={
        "expected_payment_date": "date"
    }
)


# ============================================================
# 6. CREATE FORECAST DATE RANGE
# ============================================================

start_date = min(
    daily_ar["date"].min(),
    daily_ap["date"].min()
)

end_date = max(
    daily_ar["date"].max(),
    daily_ap["date"].max()
)

dates = pd.DataFrame({
    "date": pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )
})


# ============================================================
# 7. MERGE INFLOWS AND OUTFLOWS
# ============================================================

forecast_df = dates.merge(
    daily_ar,
    on="date",
    how="left"
)

forecast_df = forecast_df.merge(
    daily_ap,
    on="date",
    how="left"
)


# ============================================================
# 8. FILL DAYS WITH NO TRANSACTIONS
# ============================================================

forecast_df["expected_inflow"] = (
    forecast_df["expected_inflow"]
    .fillna(0)
)

forecast_df["expected_outflow"] = (
    forecast_df["expected_outflow"]
    .fillna(0)
)


# ============================================================
# 9. CALCULATE NET CASH FLOW
# ============================================================

forecast_df["net_cash_flow"] = (
    forecast_df["expected_inflow"]
    - forecast_df["expected_outflow"]
)


# ============================================================
# 10. GET OPENING CASH BALANCE
# ============================================================

cash_df = cash_df.sort_values("date")

initial_cash = cash_df.iloc[0]["opening_cash"]


# ============================================================
# 11. CALCULATE PROJECTED CASH BALANCE
# ============================================================

forecast_df["projected_cash_balance"] = (
    initial_cash
    + forecast_df["net_cash_flow"].cumsum()
)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n========== DAILY CASH FLOW FORECAST ==========")

print(
    "Forecast start:",
    forecast_df["date"].min().date()
)

print(
    "Forecast end:",
    forecast_df["date"].max().date()
)

print(
    "Initial cash balance: ₹",
    round(initial_cash, 2)
)

print("\nSample forecast:")

print(
    forecast_df.head(15)
)


# ============================================================
# 13. SAVE FORECAST
# ============================================================

forecast_df.to_csv(
    "results/daily_cash_forecast.csv",
    index=False
)


print("\nSaved:")
print(
    "results/daily_cash_forecast.csv"
)