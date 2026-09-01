import joblib
import pandas as pd


# ============================================================
# 1. LOAD MODEL
# ============================================================

model = joblib.load(
    "models/payment_delay_model.pkl"
)


# ============================================================
# 2. LOAD AR INVOICES
# ============================================================

ar_df = pd.read_csv(
    "data/ar_invoices.csv"
)

ar_df["invoice_date"] = pd.to_datetime(
    ar_df["invoice_date"]
)

ar_df["due_date"] = pd.to_datetime(
    ar_df["due_date"]
)

ar_df["actual_payment_date"] = pd.to_datetime(
    ar_df["actual_payment_date"]
)


# ============================================================
# 3. SORT DATA
# ============================================================

ar_df = ar_df.sort_values(
    ["customer_id", "invoice_date"]
).reset_index(drop=True)


# ============================================================
# 4. PREDICT PAYMENT DELAY
# ============================================================

predictions = []


for index, invoice in ar_df.iterrows():

    customer_id = invoice["customer_id"]

    invoice_date = invoice["invoice_date"]

    amount = invoice["amount"]

    due_date = invoice["due_date"]


    # --------------------------------------------------------
    # Find previous invoices
    # --------------------------------------------------------

    previous_invoices = ar_df[
        (ar_df["customer_id"] == customer_id) &
        (ar_df["invoice_date"] < invoice_date)
    ]


    # --------------------------------------------------------
    # Calculate historical features
    # --------------------------------------------------------

    if previous_invoices.empty:

        previous_avg_delay = ar_df[
            "payment_delay_days"
        ].mean()

        previous_median_delay = previous_avg_delay

        previous_std_delay = 0

        previous_on_time_ratio = 0

        previous_late_ratio = 0

        days_since_previous_invoice = 0

    else:

        delays = previous_invoices[
            "payment_delay_days"
        ]

        previous_avg_delay = delays.mean()

        previous_median_delay = delays.median()

        previous_std_delay = delays.std()

        if pd.isna(previous_std_delay):
            previous_std_delay = 0


        previous_on_time_ratio = (
            (delays <= 0).sum()
            / len(delays)
        )

        previous_late_ratio = (
            (delays > 0).sum()
            / len(delays)
        )


        last_invoice_date = previous_invoices[
            "invoice_date"
        ].max()


        days_since_previous_invoice = (
            invoice_date - last_invoice_date
        ).days


    # --------------------------------------------------------
    # Create ML input
    # --------------------------------------------------------

    features = pd.DataFrame([{

        "amount": amount,

        "previous_avg_delay":
            previous_avg_delay,

        "previous_median_delay":
            previous_median_delay,

        "previous_std_delay":
            previous_std_delay,

        "previous_on_time_ratio":
            previous_on_time_ratio,

        "previous_late_ratio":
            previous_late_ratio,

        "days_since_previous_invoice":
            days_since_previous_invoice

    }])


    # --------------------------------------------------------
    # Predict delay
    # --------------------------------------------------------

    predicted_delay = model.predict(
        features
    )[0]


    # --------------------------------------------------------
    # Calculate predicted payment date
    # --------------------------------------------------------

    predicted_payment_date = (
        due_date
        + pd.Timedelta(
            days=round(predicted_delay)
        )
    )


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    predictions.append({

        "invoice_id":
            invoice["invoice_id"],

        "customer_id":
            customer_id,

        "amount":
            amount,

        "due_date":
            due_date.date(),

        "predicted_delay":
            round(predicted_delay, 2),

        "predicted_payment_date":
            predicted_payment_date.date(),

        "expected_inflow":
            amount

    })


# ============================================================
# 5. CREATE DATAFRAME
# ============================================================

expected_ar_df = pd.DataFrame(
    predictions
)


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("\n========== EXPECTED AR INFLOWS ==========")

print(
    "Number of invoices:",
    len(expected_ar_df)
)

print("\nSample predictions:")

print(
    expected_ar_df.head(10)
)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

expected_ar_df.to_csv(
    "results/expected_ar_inflows.csv",
    index=False
)


print("\nSaved:")
print(
    "results/expected_ar_inflows.csv"
)