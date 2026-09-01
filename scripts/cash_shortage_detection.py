import pandas as pd


# ============================================================
# 1. LOAD CASH UNCERTAINTY BANDS
# ============================================================

forecast_df = pd.read_csv(
    "results/cash_uncertainty_bands.csv"
)

forecast_df["date"] = pd.to_datetime(
    forecast_df["date"]
)


# ============================================================
# 2. DEFINE CASH THRESHOLDS
# ============================================================

# Minimum amount of cash we want the business
# to maintain.

WARNING_THRESHOLD = 10_000_000

CRITICAL_THRESHOLD = 5_000_000


# ============================================================
# 3. DETECT CASH RISK
# ============================================================

def classify_cash_risk(lower_cash):

    if lower_cash < CRITICAL_THRESHOLD:

        return "CRITICAL"

    elif lower_cash < WARNING_THRESHOLD:

        return "WARNING"

    else:

        return "SAFE"


forecast_df["risk_status"] = (
    forecast_df["lower_cash"]
    .apply(classify_cash_risk)
)


# ============================================================
# 4. CALCULATE CASH BUFFER
# ============================================================

forecast_df["cash_buffer"] = (
    forecast_df["lower_cash"]
    - WARNING_THRESHOLD
)


# ============================================================
# 5. FIND RISK DATES
# ============================================================

risk_df = forecast_df[
    forecast_df["risk_status"] != "SAFE"
].copy()


# ============================================================
# 6. DISPLAY SUMMARY
# ============================================================

print("\n========== CASH SHORTAGE DETECTION ==========")

print(
    "\nWarning threshold: ₹",
    WARNING_THRESHOLD
)

print(
    "Critical threshold: ₹",
    CRITICAL_THRESHOLD
)

print(
    "\nTotal forecast dates:",
    len(forecast_df)
)

print(
    "Risk dates:",
    len(risk_df)
)


# ============================================================
# 7. DISPLAY RISK BREAKDOWN
# ============================================================

print("\nRisk breakdown:")

print(
    forecast_df["risk_status"]
    .value_counts()
)


# ============================================================
# 8. DISPLAY FIRST RISK DATES
# ============================================================

if not risk_df.empty:

    print("\nFirst risk dates:")

    print(
        risk_df[
            [
                "date",
                "lower_cash",
                "base_cash",
                "upper_cash",
                "cash_buffer",
                "risk_status"
            ]
        ].head(20)
    )

else:

    print(
        "\nNo potential cash shortage detected."
    )


# ============================================================
# 9. FIND MINIMUM PROJECTED CASH
# ============================================================

minimum_cash_row = forecast_df.loc[
    forecast_df["lower_cash"].idxmin()
]


print("\nMinimum lower-case cash:")

print(
    "Date:",
    minimum_cash_row["date"].date()
)

print(
    "Lower cash: ₹",
    round(
        minimum_cash_row["lower_cash"],
        2
    )
)

print(
    "Base cash: ₹",
    round(
        minimum_cash_row["base_cash"],
        2
    )
)

print(
    "Upper cash: ₹",
    round(
        minimum_cash_row["upper_cash"],
        2
    )
)

print(
    "Risk:",
    minimum_cash_row["risk_status"]
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

forecast_df.to_csv(
    "results/cash_shortage_analysis.csv",
    index=False
)


print("\nSaved:")

print(
    "results/cash_shortage_analysis.csv"
)