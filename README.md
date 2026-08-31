# AI Finance Controller

An AI-powered finance controller designed to help businesses understand their cash position and forecast future cash flow using historical financial data.

> **Project Status: In Development**

---

## Overview

The AI Finance Controller is a financial forecasting project that combines accounts receivable (AR), accounts payable (AP), customer/vendor payment behavior, and cash-flow data.

The project is being developed step-by-step:

1. Financial data generation
2. Data validation and analysis
3. Payment behavior analysis
4. Machine learning for payment-delay prediction
5. Cash-flow forecasting
6. Cash shortage detection
7. Interactive finance dashboard

The current implementation focuses on **customer payment-delay prediction using Machine Learning**.

---

## Project Goals

The system aims to:

- Track accounts receivable (AR)
- Track accounts payable (AP)
- Analyze customer payment behavior
- Analyze vendor payment behavior
- Monitor daily cash position
- Predict customer payment delays
- Forecast future cash flow
- Identify potential cash shortages
- Provide confidence/uncertainty information
- Provide an interactive finance dashboard

---

## Current Progress

### Day 1 — Data Generation & Database

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

### Day 2 — Data Analysis & Baseline Prediction

- [x] Data validation completed
- [x] Missing-value checks completed
- [x] Duplicate ID checks completed
- [x] Invoice amount validation completed
- [x] Payment-delay calculations validated
- [x] Customer payment behavior analyzed
- [x] Vendor payment behavior analyzed
- [x] Average payment delay calculated
- [x] Median payment delay calculated
- [x] Payment-delay standard deviation calculated
- [x] Invoice count calculated
- [x] Baseline payment-date prediction created
- [x] New-customer fallback logic implemented
- [x] Uncertainty classification implemented
- [x] Baseline prediction evaluated

### Day 3 — Machine Learning

- [x] ML dataset created
- [x] Time-based train/test split created
- [x] Data leakage prevention implemented
- [x] Leakage-free customer behavior features created
- [x] Linear Regression model trained
- [x] Linear Regression evaluated
- [x] Random Forest model trained
- [x] Feature importance analyzed
- [x] Improved historical behavior features created
- [x] Improved Random Forest trained
- [x] Model performance compared
- [x] Random Forest hyperparameters tuned
- [x] Best model configuration selected
- [x] Final trained model saved
- [x] Model feature list saved
- [x] Automatic customer feature generation implemented
- [x] Payment delay prediction implemented
- [x] Predicted payment date implemented
- [x] New-customer prediction handling implemented
- [x] End-to-end testing completed
- [x] Prediction results saved

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

These patterns allow the system to analyze differences in payment behavior and provide data for the machine-learning model.

---

## Machine Learning

### ML Objective

The current ML task is:

> **Predict how many days after/before the due date a customer is expected to make payment.**

The predicted delay is then converted into an expected payment date.

### Features

The improved Random Forest model uses historical customer behavior and invoice information.

Current features include:

- `amount`
- `previous_avg_delay`
- `previous_median_delay`
- `previous_std_delay`
- `previous_on_time_ratio`
- `previous_late_ratio`
- `days_since_previous_invoice`

### Target

The target variable is:

```text
payment_delay_days