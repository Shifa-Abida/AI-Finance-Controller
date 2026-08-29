import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

customer_behavior = pd.read_csv(
    "data/customer_behavior.csv"
)

ar_df = pd.read_csv(
    "data/ar_invoices.csv"
)


# ============================================================
# 2. CALCULATE OVERALL AVERAGE DELAY
# ============================================================

overall_avg_delay = ar_df["payment_delay_days"].mean()

print("Overall average delay:", overall_avg_delay)


# ============================================================
# 3. UNCERTAINTY CLASSIFICATION
# ============================================================

def classify_uncertainty(std_delay):

    # No standard deviation means insufficient history
    if pd.isna(std_delay):
        return "High"

    if std_delay < 3:
        return "Low"

    elif std_delay < 7:
        return "Medium"

    else:
        return "High"


# ============================================================
# 4. BASELINE PREDICTION FUNCTION
# ============================================================

def predict_payment_date(customer_id, due_date):

    customer = customer_behavior[
        customer_behavior["customer_id"] == customer_id
    ]

    if customer.empty:

        # No customer history
        avg_delay = overall_avg_delay
        std_delay = None
        uncertainty = "High"

    else:

        # Use customer's historical behavior
        avg_delay = customer["avg_delay"].iloc[0]
        std_delay = customer["std_delay"].iloc[0]
        uncertainty = classify_uncertainty(std_delay)

    predicted_date = (
        pd.to_datetime(due_date)
        + pd.Timedelta(days=round(avg_delay))
    )

    return predicted_date, avg_delay, uncertainty


# ============================================================
# 5. TEST KNOWN CUSTOMER - C047
# ============================================================

prediction, avg_delay, uncertainty = predict_payment_date(
    "C047",
    "2026-09-10"
)

print("\n========== C047 PREDICTION ==========")
print("Customer: C047")
print("Due date: 2026-09-10")
print("Average delay:", round(avg_delay, 2), "days")
print("Predicted payment date:", prediction)
print("Uncertainty:", uncertainty)


# ============================================================
# 6. TEST UNCERTAIN CUSTOMER - C009
# ============================================================

prediction, avg_delay, uncertainty = predict_payment_date(
    "C009",
    "2026-09-10"
)

print("\n========== C009 PREDICTION ==========")
print("Customer: C009")
print("Due date: 2026-09-10")
print("Average delay:", round(avg_delay, 2), "days")
print("Predicted payment date:", prediction)
print("Uncertainty:", uncertainty)


# ============================================================
# 7. TEST NEW CUSTOMER - C999
# ============================================================

prediction, avg_delay, uncertainty = predict_payment_date(
    "C999",
    "2026-09-10"
)

print("\n========== C999 PREDICTION ==========")
print("Customer: C999")
print("Due date: 2026-09-10")
print("Average delay:", round(avg_delay, 2), "days")
print("Predicted payment date:", prediction)
print("Uncertainty:", uncertainty)

# ============================================================
# 5. EVALUATE BASELINE PREDICTIONS
# ============================================================

def evaluate_baseline():

    results = []

    for _, invoice in ar_df.iterrows():

        predicted_date, avg_delay, uncertainty = predict_payment_date(
            invoice["customer_id"],
            invoice["due_date"]
        )

        actual_date = pd.to_datetime(
            invoice["actual_payment_date"]
        )

        error_days = abs(
            (actual_date - predicted_date).days
        )

        results.append({
            "invoice_id": invoice["invoice_id"],
            "customer_id": invoice["customer_id"],
            "due_date": invoice["due_date"],
            "actual_payment_date": actual_date,
            "predicted_payment_date": predicted_date,
            "error_days": error_days,
            "uncertainty": uncertainty
        })

    return pd.DataFrame(results)

# ============================================================
# 6. CALCULATE BASELINE MAE
# ============================================================

baseline_results = evaluate_baseline()

baseline_mae = baseline_results["error_days"].mean()

print("\n========== BASELINE EVALUATION ==========")
print("Number of invoices evaluated:", len(baseline_results))
print("Baseline MAE:", round(baseline_mae, 2), "days")

print("\n========== SAMPLE PREDICTIONS ==========")

print(
    baseline_results[
        [
            "invoice_id",
            "customer_id",
            "due_date",
            "actual_payment_date",
            "predicted_payment_date",
            "error_days",
            "uncertainty"
        ]
    ].head(10)
)