import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


# ============================================================
# 1. CONFIGURATION
# ============================================================

NUM_CUSTOMERS = 50
NUM_VENDORS = 30

NUM_AR_INVOICES = 700
NUM_AP_INVOICES = 300

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)

# Makes the generated dataset reproducible
random.seed(42)
np.random.seed(42)


# ============================================================
# 2. CUSTOMER BEHAVIOR TYPES
# ============================================================

behavior_types = [
    "ON_TIME",
    "CONSISTENT_LATE",
    "VERY_LATE",
    "UNPREDICTABLE"
]


# ============================================================
# 3. GENERATE CUSTOMERS
# ============================================================

customers = []

for i in range(1, NUM_CUSTOMERS + 1):

    customer_id = f"C{i:03d}"
    customer_name = f"Customer {i}"

    behavior_type = random.choice(behavior_types)

    customers.append({
        "customer_id": customer_id,
        "customer_name": customer_name,
        "behavior_type": behavior_type
    })

customers_df = pd.DataFrame(customers)


# ============================================================
# 4. GENERATE VENDORS
# ============================================================

vendors = []

for i in range(1, NUM_VENDORS + 1):

    vendor_id = f"V{i:03d}"
    vendor_name = f"Vendor {i}"

    behavior_type = random.choice(behavior_types)

    vendors.append({
        "vendor_id": vendor_id,
        "vendor_name": vendor_name,
        "behavior_type": behavior_type
    })

vendors_df = pd.DataFrame(vendors)


# ============================================================
# 5. GENERATE AR INVOICES
# ============================================================

ar_invoices = []

days_range = (END_DATE - START_DATE).days

for i in range(1, NUM_AR_INVOICES + 1):

    # Select a random customer
    customer = random.choice(customers)

    customer_id = customer["customer_id"]
    behavior = customer["behavior_type"]

    # Generate invoice date
    random_days = random.randint(0, days_range)

    invoice_date = START_DATE + timedelta(days=random_days)

    # Due date = 30 days after invoice
    due_date = invoice_date + timedelta(days=30)

    # --------------------------------------------------------
    # Seasonal invoice amount
    # --------------------------------------------------------

    month = invoice_date.month

    if month in [10, 11, 12]:
        seasonal_multiplier = 1.3

    elif month in [3, 6, 9]:
        seasonal_multiplier = 1.15

    else:
        seasonal_multiplier = 1.0

    amount = int(
        random.randint(20000, 500000)
        * seasonal_multiplier
    )

    # --------------------------------------------------------
    # Payment delay based on customer behavior
    # --------------------------------------------------------

    if behavior == "ON_TIME":
        delay = random.randint(-2, 2)

    elif behavior == "CONSISTENT_LATE":
        delay = random.randint(2, 5)

    elif behavior == "VERY_LATE":
        delay = random.randint(8, 15)

    else:
        # UNPREDICTABLE
        delay = random.randint(-5, 30)

    # Actual payment date
    actual_payment_date = due_date + timedelta(days=delay)

    ar_invoices.append({
        "invoice_id": f"AR{i:04d}",
        "customer_id": customer_id,
        "invoice_date": invoice_date.date(),
        "due_date": due_date.date(),
        "amount": amount,
        "actual_payment_date": actual_payment_date.date(),
        "payment_delay_days": delay
    })


ar_df = pd.DataFrame(ar_invoices)


# ============================================================
# 6. GENERATE AP INVOICES
# ============================================================

ap_invoices = []

for i in range(1, NUM_AP_INVOICES + 1):

    # Select a random vendor
    vendor = random.choice(vendors)

    vendor_id = vendor["vendor_id"]
    behavior = vendor["behavior_type"]

    # Generate invoice date
    random_days = random.randint(0, days_range)

    invoice_date = START_DATE + timedelta(days=random_days)

    # Due date = 30 days after invoice
    due_date = invoice_date + timedelta(days=30)

    # --------------------------------------------------------
    # Seasonal invoice amount
    # --------------------------------------------------------

    month = invoice_date.month

    if month in [10, 11, 12]:
        seasonal_multiplier = 1.3

    elif month in [3, 6, 9]:
        seasonal_multiplier = 1.15

    else:
        seasonal_multiplier = 1.0

    amount = int(
        random.randint(10000, 300000)
        * seasonal_multiplier
    )

    # --------------------------------------------------------
    # Payment delay based on vendor behavior
    # --------------------------------------------------------

    if behavior == "ON_TIME":
        delay = random.randint(-2, 2)

    elif behavior == "CONSISTENT_LATE":
        delay = random.randint(2, 5)

    elif behavior == "VERY_LATE":
        delay = random.randint(8, 15)

    else:
        # UNPREDICTABLE
        delay = random.randint(-5, 30)

    # Actual payment date
    actual_payment_date = due_date + timedelta(days=delay)

    ap_invoices.append({
        "invoice_id": f"AP{i:04d}",
        "vendor_id": vendor_id,
        "invoice_date": invoice_date.date(),
        "due_date": due_date.date(),
        "amount": amount,
        "actual_payment_date": actual_payment_date.date(),
        "payment_delay_days": delay
    })


ap_df = pd.DataFrame(ap_invoices)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n========== CUSTOMERS ==========")
print(customers_df.head())

print("\n========== VENDORS ==========")
print(vendors_df.head())

print("\n========== AR INVOICES ==========")
print(ar_df.head())

print("\n========== AP INVOICES ==========")
print(ap_df.head())


# ============================================================
# 8. BASIC STATISTICS
# ============================================================

print("\n========== BASIC STATISTICS ==========")

print("Number of customers:", len(customers_df))
print("Number of vendors:", len(vendors_df))

print("Number of AR invoices:", len(ar_df))
print("Number of AP invoices:", len(ap_df))

print("Total transactions:", len(ar_df) + len(ap_df))

print("\nTotal AR: ₹", f"{ar_df['amount'].sum():,.2f}")
print("Total AP: ₹", f"{ap_df['amount'].sum():,.2f}")

print(
    "\nAverage customer payment delay:",
    round(ar_df["payment_delay_days"].mean(), 2),
    "days"
)

print(
    "Average vendor payment delay:",
    round(ap_df["payment_delay_days"].mean(), 2),
    "days"
)

# ============================================================
# 7. GENERATE DAILY CASH BALANCE
# ============================================================

cash_balance = []

current_cash = 10_000_000  # Starting cash: ₹1 crore

current_date = START_DATE

while current_date <= END_DATE:

    # AR payments received on this date
    ar_inflow = ar_df[
        ar_df["actual_payment_date"] == current_date.date()
    ]["amount"].sum()

    # AP payments made on this date
    ap_outflow = ap_df[
        ap_df["actual_payment_date"] == current_date.date()
    ]["amount"].sum()

    opening_cash = current_cash

    closing_cash = (
        opening_cash
        + ar_inflow
        - ap_outflow
    )

    cash_balance.append({
        "date": current_date.date(),
        "opening_cash": round(opening_cash, 2),
        "cash_inflow": round(ar_inflow, 2),
        "cash_outflow": round(ap_outflow, 2),
        "closing_cash": round(closing_cash, 2)
    })

    current_cash = closing_cash

    current_date += timedelta(days=1)


cash_df = pd.DataFrame(cash_balance)


print("\n========== CASH BALANCE ==========")
print(cash_df.head())

# ============================================================
# 8. SAVE DATA AS CSV FILES
# ============================================================

customers_df.to_csv("data/customers.csv", index=False)
vendors_df.to_csv("data/vendors.csv", index=False)
ar_df.to_csv("data/ar_invoices.csv", index=False)
ap_df.to_csv("data/ap_invoices.csv", index=False)
cash_df.to_csv("data/cash_balance.csv", index=False)

print("\n========== CSV FILES SAVED ==========")
print("All files saved successfully in the data/ folder.")

# ============================================================
# 9. DATA VALIDATION
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