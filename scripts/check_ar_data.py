import pandas as pd

ar_df = pd.read_csv("data/ar_invoices.csv")

print(ar_df.columns)

print("\nDate range:")
print("Invoice date:", ar_df["invoice_date"].min(), "→", ar_df["invoice_date"].max())
print("Due date:", ar_df["due_date"].min(), "→", ar_df["due_date"].max())
print("Payment date:", ar_df["actual_payment_date"].min(), "→", ar_df["actual_payment_date"].max())

print("\nLast 10 invoices:")
print(
    ar_df[
        [
            "invoice_id",
            "customer_id",
            "invoice_date",
            "due_date",
            "amount",
            "actual_payment_date",
            "payment_delay_days"
        ]
    ].tail(10)
)