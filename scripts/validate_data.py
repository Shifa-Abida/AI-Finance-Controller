import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

customers_df = pd.read_csv("data/customers.csv")
vendors_df = pd.read_csv("data/vendors.csv")
ar_df = pd.read_csv("data/ar_invoices.csv")
ap_df = pd.read_csv("data/ap_invoices.csv")
cash_df = pd.read_csv("data/cash_balance.csv")

# ============================================================
# 2. DATA VALIDATION
# ============================================================

print("\n========== DATA VALIDATION ==========")

# Row counts
print("\nRow counts:")
print("Customers:", len(customers_df))
print("Vendors:", len(vendors_df))
print("AR invoices:", len(ar_df))
print("AP invoices:", len(ap_df))
print("Cash balance:", len(cash_df))


# Check duplicate IDs
print("\nDuplicate ID checks:")
print("Duplicate customer IDs:", customers_df["customer_id"].duplicated().sum())
print("Duplicate vendor IDs:", vendors_df["vendor_id"].duplicated().sum())
print("Duplicate AR invoice IDs:", ar_df["invoice_id"].duplicated().sum())
print("Duplicate AP invoice IDs:", ap_df["invoice_id"].duplicated().sum())


# Check missing values
print("\nMissing values:")

print("Customers:")
print(customers_df.isnull().sum())

print("\nVendors:")
print(vendors_df.isnull().sum())

print("\nAR invoices:")
print(ar_df.isnull().sum())

print("\nAP invoices:")
print(ap_df.isnull().sum())

print("\nCash balance:")
print(cash_df.isnull().sum())


# Check invoice amounts
print("\nInvalid invoice amounts:")
print("AR <= 0:", (ar_df["amount"] <= 0).sum())
print("AP <= 0:", (ap_df["amount"] <= 0).sum())


# Check payment delay calculation
print("\nPayment delay validation:")

ar_delay_check = (
    pd.to_datetime(ar_df["actual_payment_date"])
    - pd.to_datetime(ar_df["due_date"])
).dt.days

ap_delay_check = (
    pd.to_datetime(ap_df["actual_payment_date"])
    - pd.to_datetime(ap_df["due_date"])
).dt.days

print(
    "AR incorrect delays:",
    (ar_delay_check != ar_df["payment_delay_days"]).sum()
)

print(
    "AP incorrect delays:",
    (ap_delay_check != ap_df["payment_delay_days"]).sum()
)
