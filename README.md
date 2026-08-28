# AI Finance Controller

An AI-powered finance controller designed to help businesses understand their cash position and eventually forecast future cash flow using historical financial data.

> 🚧 **Project Status: In Development**

## 📌 Overview

The AI Finance Controller is a financial forecasting project that combines transaction data, customer/vendor payment behavior, and cash-flow information to build a system capable of predicting future cash positions.

The project is being developed step-by-step, starting with synthetic financial data and a PostgreSQL database, followed by data analysis, feature engineering, forecasting models, and eventually an interactive dashboard.

---

## 🎯 Project Goals

The system aims to:

- Track accounts receivable (AR)
- Track accounts payable (AP)
- Analyze customer payment behavior
- Analyze vendor payment behavior
- Monitor daily cash position
- Forecast future cash flow
- Identify potential cash shortages
- Provide confidence ranges around forecasts
- Eventually provide an interactive finance dashboard

---

## 🏗️ Current Progress

### Completed

- [x] Project structure created
- [x] PostgreSQL database created
- [x] Database schema designed
- [x] Synthetic financial data generator created
- [x] 50 customers generated
- [x] 30 vendors generated
- [x] 700 AR invoices generated
- [x] 300 AP invoices generated
- [x] 1,000 total financial transactions generated
- [x] 365 days of cash-balance data generated
- [x] Data validation completed
- [x] CSV files generated
- [x] CSV data imported into PostgreSQL

### In Progress

- [ ] SQL-based financial analysis
- [ ] Feature engineering
- [ ] Cash-flow forecasting
- [ ] Machine learning model
- [ ] Forecast confidence intervals
- [ ] Cash shortage detection
- [ ] Finance dashboard
- [ ] API integration

---

## 🗃️ Current Dataset

The project currently uses synthetic financial data.

| Dataset | Records |
|---|---:|
| Customers | 50 |
| Vendors | 30 |
| AR Invoices | 700 |
| AP Invoices | 300 |
| Total Transactions | 1,000 |
| Daily Cash Balance | 365 days |

### Customer & Vendor Payment Behaviors

The synthetic dataset contains different payment behavior patterns:

- `ON_TIME`
- `CONSISTENT_LATE`
- `VERY_LATE`
- `UNPREDICTABLE`

This allows the future forecasting model to learn patterns in payment behavior.

---

## 🛠️ Current Tech Stack

- Python
- Pandas
- NumPy
- PostgreSQL
- DBeaver
- Git & GitHub

> Additional technologies will be added to this section as they are actually implemented.

---

## 📁 Project Structure

```text
AI-Finance-Controller/
│
├── data/
│   ├── customers.csv
│   ├── vendors.csv
│   ├── ar_invoices.csv
│   ├── ap_invoices.csv
│   └── cash_balance.csv
│
├── database/
│
├── scripts/
│   └── generate_data.py
│
├── notebooks/
│
└── README.md