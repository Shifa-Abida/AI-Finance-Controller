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


# ============================================================
# 3. PREDICTION FUNCTION
# ============================================================

def predict_payment(customer_id, invoice_amount, due_date):

    due_date = pd.to_datetime(due_date)

    # --------------------------------------------------------
    # Find previous invoices
    # --------------------------------------------------------

    previous_invoices = ar_df[
        (ar_df["customer_id"] == customer_id) &
        (ar_df["invoice_date"] < due_date)
    ].sort_values("invoice_date")


    # --------------------------------------------------------
    # Calculate features
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
            due_date - last_invoice_date
        ).days


    # --------------------------------------------------------
    # Create ML input
    # --------------------------------------------------------

    features = pd.DataFrame([{

        "amount": invoice_amount,

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
    # Predict
    # --------------------------------------------------------

    predicted_delay = model.predict(
        features
    )[0]


    # --------------------------------------------------------
    # Payment date
    # --------------------------------------------------------

    predicted_payment_date = (
        due_date
        + pd.Timedelta(
            days=round(predicted_delay)
        )
    )


    return {
        "customer_id": customer_id,
        "invoice_amount": invoice_amount,
        "due_date": due_date.date(),
        "previous_invoices": len(previous_invoices),
        "historical_avg_delay": round(
            previous_avg_delay, 2
        ),
        "predicted_delay": round(
            predicted_delay, 2
        ),
        "predicted_payment_date":
            predicted_payment_date.date()
    }


# ============================================================
# 4. TEST CASES
# ============================================================

test_cases = [

    # Existing customer with relatively predictable behavior
    {
        "customer_id": "C047",
        "invoice_amount": 300000,
        "due_date": "2026-09-10"
    },

    # Existing customer with unpredictable behavior
    {
        "customer_id": "C009",
        "invoice_amount": 300000,
        "due_date": "2026-09-10"
    },

    # Completely new customer
    {
        "customer_id": "C999",
        "invoice_amount": 300000,
        "due_date": "2026-09-10"
    }
]


# ============================================================
# 5. RUN TESTS
# ============================================================

print("\n========== END-TO-END TESTING ==========")

for case in test_cases:

    result = predict_payment(
        case["customer_id"],
        case["invoice_amount"],
        case["due_date"]
    )

    print("\n----------------------------------------")

    print(
        "Customer:",
        result["customer_id"]
    )

    print(
        "Invoice amount: ₹",
        result["invoice_amount"]
    )

    print(
        "Due date:",
        result["due_date"]
    )

    print(
        "Previous invoices:",
        result["previous_invoices"]
    )

    print(
        "Historical average delay:",
        result["historical_avg_delay"],
        "days"
    )

    print(
        "Predicted delay:",
        result["predicted_delay"],
        "days"
    )

    print(
        "Predicted payment date:",
        result["predicted_payment_date"]
    )

    # ============================================================
# 6. SAVE PREDICTIONS
# ============================================================

results = []

for case in test_cases:

    result = predict_payment(
        case["customer_id"],
        case["invoice_amount"],
        case["due_date"]
    )

    results.append(result)


results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/payment_predictions.csv",
    index=False
)

print("\n========================================")
print("Predictions saved:")
print("results/payment_predictions.csv")