# e-commerce-order-prediction-user-analytics
This project builds an end-to-end data science pipeline to predict user repurchase behavior and analyze shopping patterns on an e-commerce platform.

## Goals
### 1. Predict Next-7-Day Orders

Build a machine-learning model to predict whether a user will place a new order within 7 days of their most recent purchase, targeting an ROC-AUC ≥ 0.75 on the held-out test set.

Understand Key Retention Drivers
Identify which user behaviors (e.g., order frequency, basket size, preferred departments) most influence repurchase, using feature-importance and SHAP analysis.

User segmentation and RFM metrics: Model predictions and explanations, including a ranked list of high-risk churn users.


### 2. Next-Basket Recommendation

Build a recommendation model that predicts the set of products a user is most likely to purchase in their next order, based on past shopping history.

Generate a personalized ranked list of candidate products for each user with a target top-k hit rate (e.g., Hit@10 ≥ 0.35 on the held-out test set).
Analyze purchase frequency, basket size, and product co-occurrence patterns to understand key factors driving repeat purchases and cross-selling opportunities.

## Dataset
Instacart Market Basket Analysis (Kaggle)
https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset/data
Download the public CSV files from Kaggle and store them in a version-controlled GitHub repository.
### Tables used
orders.csv – order metadata (user ID, order sequence, day-of-week, hour-of-day, days-since-prior-order)

order_products__prior.csv & order_products__train.csv – product IDs and quantities for each order

products.csv, aisles.csv, departments.csv – product and category details

## Model
### 1. Prediction Task

**Goal** Predict whether a user will place a new order within 7 days of their most recent purchase (binary classification).

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

### 2. Next-Basket Recommendation
**Data Preparation**
Use orders and order_products__prior/train to construct a user-to-time ordered "shopping basket sequence"

For each user, leave out the last order as the test basket (leave-one-basket-out), use the penultimate order for validation, and use the remaining orders for training

Negative Sampling: Sample unpurchased items for each positive item as negative examples (for learning to rank)

**Two-Stage Recommendation Framework**

1) Candidate Recall (Recall):

Collaborative Filtering (Item-Item CF): Find similar items based on co-occurrence/cosine similarity

Association Rules (Apriori / FPGrowth): Generate {A, B}→C rules as recall candidates

Matrix Factorization (Implicit ALS: Obtaining Top-N Candidates Using Implicit Feedback-Based MF

2) Ranking:

Train a Gradient-Boosted Decision Tree (e.g., LightGBM/XGBoost with pairwise ranking loss) to re-score the candidates.

**Evaluation Metrics**
Top-K Hit Rate (Hit@K), Recall@K, Precision@K, Mean Average Precision (MAP), NDCG

## Data Visualization

For Next-Order Prediction
ROC and Precision–Recall curves, confusion matrix heatmap, and feature-importance bar charts (e.g., SHAP values).

For Next-Basket Recommendation
Precision@K and Recall@K curves.
Hit-rate vs. K comparison bar chart.

(Optional: Personalized Demo)
Input a user ID to view their past baskets, the model’s top-k predicted items, and SHAP-style explanations for top recommendations.
