# AI Finance Controller

An AI-powered finance controller designed to help businesses understand their cash position and eventually forecast future cash flow using historical financial data.

> **Project Status: In Development**

## Overview

The AI Finance Controller is a financial forecasting project that combines transaction data, customer/vendor payment behavior, and cash-flow information to build a system capable of predicting future cash positions.

The project is being developed step-by-step, starting with synthetic financial data and a PostgreSQL database, followed by data validation, payment behavior analysis, baseline forecasting, feature engineering, machine learning models, and eventually an interactive dashboard.

---

## Project Goals

The system aims to:

- Track accounts receivable (AR)
- Track accounts payable (AP)
- Analyze customer payment behavior
- Analyze vendor payment behavior
- Monitor daily cash position
- Forecast future cash flow
- Predict customer payment dates
- Identify potential cash shortages
- Provide confidence ranges around forecasts
- Eventually provide an interactive finance dashboard

---

## Current Progress

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
- [x] CSV files generated
- [x] CSV data imported into PostgreSQL
- [x] Data validation completed
- [x] Customer payment behavior analyzed
- [x] Vendor payment behavior analyzed
- [x] Customer behavior dataset generated
- [x] Vendor behavior dataset generated
- [x] Baseline payment-date prediction created
- [x] New-customer fallback implemented
- [x] Prediction uncertainty classification implemented
- [x] Baseline evaluated on 700 AR invoices
- [x] Baseline MAE calculated: **2.33 days**

### In Progress

- [ ] Feature engineering
- [ ] Machine learning model
- [ ] Proper time-based train/test evaluation
- [ ] Improve payment-date prediction
- [ ] Cash-flow forecasting
- [ ] Forecast confidence intervals
- [ ] Cash shortage detection
- [ ] Finance dashboard
- [ ] API integration

---

## Current Dataset

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

These patterns allow the system to analyze historical payment behavior and eventually provide features for machine learning models.

---

## Payment Behavior Analysis

For each customer and vendor, the system calculates:

- Average payment delay
- Median payment delay
- Standard deviation of payment delay
- Number of invoices

Example customer behavior:

```text
Customer: C047

Average delay:       12.29 days
Median delay:        12 days
Standard deviation:  2.22 days
Invoice count:       21