import pandas as pd


# ============================================================
# 1. LOAD AP INVOICES
# ============================================================

ap_df = pd.read_csv(
    "data/ap_invoices.csv"
)


# ============================================================
# 2. CONVERT DATE COLUMNS
# ============================================================

ap_df["invoice_date"] = pd.to_datetime(
    ap_df["invoice_date"]
)

ap_df["due_date"] = pd.to_datetime(
    ap_df["due_date"]
)

ap_df["actual_payment_date"] = pd.to_datetime(
    ap_df["actual_payment_date"]
)


# ============================================================
# 3. SORT DATA
# ============================================================

ap_df = ap_df.sort_values(
    ["vendor_id", "invoice_date"]
).reset_index(drop=True)


# ============================================================
# 4. GENERATE EXPECTED AP OUTFLOWS
# ============================================================

predictions = []


for index, invoice in ap_df.iterrows():

    vendor_id = invoice["vendor_id"]

    invoice_date = invoice["invoice_date"]

    amount = invoice["amount"]

    due_date = invoice["due_date"]


    # --------------------------------------------------------
    # Find previous invoices for this vendor
    # --------------------------------------------------------

    previous_invoices = ap_df[
        (ap_df["vendor_id"] == vendor_id) &
        (ap_df["invoice_date"] < invoice_date)
    ]


    # --------------------------------------------------------
    # Calculate vendor historical behavior
    # --------------------------------------------------------

    if previous_invoices.empty:

        # New vendor:
        # use overall historical average

        historical_avg_delay = ap_df[
            "payment_delay_days"
        ].mean()

    else:

        historical_avg_delay = (
            previous_invoices[
                "payment_delay_days"
            ].mean()
        )


    # --------------------------------------------------------
    # Expected payment delay
    # --------------------------------------------------------

    expected_delay = round(
        historical_avg_delay,
        2
    )


    # --------------------------------------------------------
    # Expected payment date
    # --------------------------------------------------------

    expected_payment_date = (
        due_date
        + pd.Timedelta(
            days=round(expected_delay)
        )
    )


    # --------------------------------------------------------
    # Expected cash outflow
    # --------------------------------------------------------

    expected_outflow = amount


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    predictions.append({

        "invoice_id":
            invoice["invoice_id"],

        "vendor_id":
            vendor_id,

        "amount":
            amount,

        "due_date":
            due_date.date(),

        "expected_delay":
            expected_delay,

        "expected_payment_date":
            expected_payment_date.date(),

        "expected_outflow":
            expected_outflow

    })


# ============================================================
# 5. CREATE DATAFRAME
# ============================================================

expected_ap_df = pd.DataFrame(
    predictions
)


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("\n========== EXPECTED AP OUTFLOWS ==========")

print(
    "Number of invoices:",
    len(expected_ap_df)
)

print("\nSample predictions:")

print(
    expected_ap_df.head(10)
)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

expected_ap_df.to_csv(
    "results/expected_ap_outflows.csv",
    index=False
)


print("\nSaved:")

print(
    "results/expected_ap_outflows.csv"
)