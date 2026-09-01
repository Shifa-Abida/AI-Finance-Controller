import pandas as pd
import math


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
# 3. LOAD AP HISTORICAL DATA
# ============================================================

historical_ap = pd.read_csv(
    "data/ap_invoices.csv"
)

ap_delay_std = historical_ap[
    "payment_delay_days"
].std()


# ============================================================
# 4. AR MODEL UNCERTAINTY
# ============================================================

# Day 3 model MAE
# This represents the average prediction error in days.

ar_mae = 2.33


# Convert uncertainty into whole days

ar_uncertainty_days = math.ceil(
    ar_mae
)

ap_uncertainty_days = math.ceil(
    ap_delay_std
)


print("\n========== UNCERTAINTY SETTINGS ==========")

print(
    "AR prediction uncertainty:",
    ar_mae,
    "days"
)

print(
    "AR scenario shift:",
    ar_uncertainty_days,
    "days"
)

print(
    "AP historical delay std:",
    round(ap_delay_std, 2),
    "days"
)

print(
    "AP scenario shift:",
    ap_uncertainty_days,
    "days"
)


# ============================================================
# 5. CREATE BASE CASE
# ============================================================

base_ar = (
    ar_df
    .groupby("predicted_payment_date")["expected_inflow"]
    .sum()
    .reset_index()
)

base_ar = base_ar.rename(
    columns={
        "predicted_payment_date": "date"
    }
)


base_ap = (
    ap_df
    .groupby("expected_payment_date")["expected_outflow"]
    .sum()
    .reset_index()
)

base_ap = base_ap.rename(
    columns={
        "expected_payment_date": "date"
    }
)


# ============================================================
# 6. CREATE LOWER-CASE SCENARIO
# ============================================================

# Lower cash means:
#
# AR payments arrive later
# AP payments leave earlier

lower_ar = ar_df.copy()

lower_ar["date"] = (
    lower_ar["predicted_payment_date"]
    + pd.to_timedelta(
        ar_uncertainty_days,
        unit="D"
    )
)

lower_ar = (
    lower_ar
    .groupby("date")["expected_inflow"]
    .sum()
    .reset_index()
)


lower_ap = ap_df.copy()

lower_ap["date"] = (
    lower_ap["expected_payment_date"]
    - pd.to_timedelta(
        ap_uncertainty_days,
        unit="D"
    )
)

lower_ap = (
    lower_ap
    .groupby("date")["expected_outflow"]
    .sum()
    .reset_index()
)


# ============================================================
# 7. CREATE UPPER-CASE SCENARIO
# ============================================================

# Upper cash means:
#
# AR payments arrive earlier
# AP payments leave later

upper_ar = ar_df.copy()

upper_ar["date"] = (
    upper_ar["predicted_payment_date"]
    - pd.to_timedelta(
        ar_uncertainty_days,
        unit="D"
    )
)

upper_ar = (
    upper_ar
    .groupby("date")["expected_inflow"]
    .sum()
    .reset_index()
)


upper_ap = ap_df.copy()

upper_ap["date"] = (
    upper_ap["expected_payment_date"]
    + pd.to_timedelta(
        ap_uncertainty_days,
        unit="D"
    )
)

upper_ap = (
    upper_ap
    .groupby("date")["expected_outflow"]
    .sum()
    .reset_index()
)


# ============================================================
# 8. CREATE COMPLETE DATE RANGE
# ============================================================

all_dates = pd.concat([
    base_ar[["date"]],
    base_ap[["date"]],
    lower_ar[["date"]],
    lower_ap[["date"]],
    upper_ar[["date"]],
    upper_ap[["date"]]
])


dates = pd.DataFrame({
    "date": pd.date_range(
        start=all_dates["date"].min(),
        end=all_dates["date"].max(),
        freq="D"
    )
})


# ============================================================
# 9. MERGE BASE CASE
# ============================================================

forecast = dates.merge(
    base_ar,
    on="date",
    how="left"
)

forecast = forecast.merge(
    base_ap,
    on="date",
    how="left"
)

forecast = forecast.rename(
    columns={
        "expected_inflow":
            "base_inflow",

        "expected_outflow":
            "base_outflow"
    }
)


# ============================================================
# 10. MERGE LOWER CASE
# ============================================================

forecast = forecast.merge(
    lower_ar.rename(
        columns={
            "expected_inflow":
                "lower_inflow"
        }
    ),
    on="date",
    how="left"
)

forecast = forecast.merge(
    lower_ap.rename(
        columns={
            "expected_outflow":
                "lower_outflow"
        }
    ),
    on="date",
    how="left"
)


# ============================================================
# 11. MERGE UPPER CASE
# ============================================================

forecast = forecast.merge(
    upper_ar.rename(
        columns={
            "expected_inflow":
                "upper_inflow"
        }
    ),
    on="date",
    how="left"
)

forecast = forecast.merge(
    upper_ap.rename(
        columns={
            "expected_outflow":
                "upper_outflow"
        }
    ),
    on="date",
    how="left"
)


# ============================================================
# 12. FILL MISSING VALUES
# ============================================================

cash_columns = [
    "base_inflow",
    "base_outflow",
    "lower_inflow",
    "lower_outflow",
    "upper_inflow",
    "upper_outflow"
]

forecast[cash_columns] = (
    forecast[cash_columns]
    .fillna(0)
)


# ============================================================
# 13. CALCULATE NET CASH FLOW
# ============================================================

forecast["base_net_cash_flow"] = (
    forecast["base_inflow"]
    - forecast["base_outflow"]
)


forecast["lower_net_cash_flow"] = (
    forecast["lower_inflow"]
    - forecast["lower_outflow"]
)


forecast["upper_net_cash_flow"] = (
    forecast["upper_inflow"]
    - forecast["upper_outflow"]
)


# ============================================================
# 14. INITIAL CASH
# ============================================================

initial_cash = 10_000_000


# ============================================================
# 15. CALCULATE CASH BALANCES
# ============================================================

forecast["base_cash"] = (
    initial_cash
    + forecast[
        "base_net_cash_flow"
    ].cumsum()
)


forecast["lower_cash"] = (
    initial_cash
    + forecast[
        "lower_net_cash_flow"
    ].cumsum()
)


forecast["upper_cash"] = (
    initial_cash
    + forecast[
        "upper_net_cash_flow"
    ].cumsum()
)


# ============================================================
# 16. DISPLAY RESULTS
# ============================================================

print("\n========== CASH UNCERTAINTY BANDS ==========")

print(
    "\nSample uncertainty forecast:"
)

print(
    forecast[
        [
            "date",
            "lower_cash",
            "base_cash",
            "upper_cash"
        ]
    ].head(15)
)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

forecast[
    [
        "date",
        "lower_cash",
        "base_cash",
        "upper_cash"
    ]
].to_csv(
    "results/cash_uncertainty_bands.csv",
    index=False
)


print("\nSaved:")

print(
    "results/cash_uncertainty_bands.csv"
)