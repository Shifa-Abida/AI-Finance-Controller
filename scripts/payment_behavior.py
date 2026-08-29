import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

ar_df = pd.read_csv("data/ar_invoices.csv")
ap_df = pd.read_csv("data/ap_invoices.csv")


# ============================================================
# 2. CUSTOMER PAYMENT BEHAVIOR
# ============================================================

customer_behavior = ar_df.groupby("customer_id")["payment_delay_days"].agg(
    avg_delay="mean",
    median_delay="median",
    std_delay="std",
    invoice_count="count"
).reset_index()


# ============================================================
# 3. VENDOR PAYMENT BEHAVIOR
# ============================================================

vendor_behavior = ap_df.groupby("vendor_id")["payment_delay_days"].agg(
    avg_delay="mean",
    median_delay="median",
    std_delay="std",
    invoice_count="count"
).reset_index()


# ============================================================
# 4. DISPLAY RESULTS
# ============================================================

print("\n========== CUSTOMER PAYMENT BEHAVIOR ==========")
print(customer_behavior)

print("\n========== VENDOR PAYMENT BEHAVIOR ==========")
print(vendor_behavior)

# ============================================================
# 5. SAVE BEHAVIOR DATA
# ============================================================

customer_behavior.to_csv(
    "data/customer_behavior.csv",
    index=False
)

vendor_behavior.to_csv(
    "data/vendor_behavior.csv",
    index=False
)

print("\nBehavior data saved successfully.")