# e-commerce-order-prediction-user-analytics
The goal of this project is to build a machine learning model that predicts whether a user will place a new order within 7 days of their most recent purchase. This task addresses short-term user retention and helps identify high-risk churn users.
## Goals
### 1. Predict Next-7-Day Orders

Build a machine-learning model to predict whether a user will place a new order within 7 days of their most recent purchase.

Understand Key Retention Drivers
Identify which user behaviors (e.g., order frequency, basket size, preferred departments) most influence repurchase, using feature-importance and SHAP analysis.

User segmentation and RFM metrics: Model predictions and explanations, including a ranked list of high-risk churn users.

## Dataset
Instacart Market Basket Analysis (Kaggle)
https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset/data
Download the public CSV files from Kaggle and store them in a version-controlled GitHub repository.

Data cleaning: handle missing values (e.g., days_since_prior_order for the first order), merge product metadata, ensure chronological ordering of user histories.
### Tables used
orders.csv – order metadata (user ID, order sequence, day-of-week, hour-of-day, days-since-prior-order)

order_products__prior.csv & order_products__train.csv – product IDs and quantities for each order

products.csv, aisles.csv, departments.csv – product and category details

## Model
### 1. Prediction Task

**Goal** Predict whether a user will place a new order within 7 days of their most recent purchase (binary classification). The 7-day window is defined as the period immediately following an order, i.e., if the next purchase occurs within 0–6 days after the last order (same-day repeat orders included), it is labeled as a positive case; orders on day 7 or later are labeled as negative.

Feature Engineering

User-level: average days between orders, total orders, basket size, preferred departments.

Order-level: order_dow (day of week), order_hour_of_day, days_since_prior_order.

Product aggregation: diversity of departments, top-N frequently purchased items.

**Models to Train**

Baseline: Logistic Regression for an interpretable starting point.

Tree-Based Models: Random Forest and XGBoost (Gradient Boosted Trees) to capture non-linear interactions and typically strong tabular performance.

(Optional Advanced): LightGBM or CatBoost for comparison and possible speed/accuracy gains.

**Training & Validation**

Temporal split: first 70 % of the timeline for training, next 15 % for validation, final 15 % for testing to avoid data leakage.

Hyper-parameter tuning via grid search or Bayesian optimization.

**Evaluation Metrics**

Primary: ROC-AUC and PR-AUC (handle class imbalance).

Secondary: Accuracy, Precision, Recall, F1-score.


## Data Visualization

ROC and Precision–Recall curves, confusion matrix heatmap, and feature-importance bar charts (e.g., SHAP values).

Instead of setting a fixed target (e.g., ROC-AUC ≥ 0.75), performance will be compared across baselines and optimized models to assess improvements.

