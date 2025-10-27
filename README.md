# e-commerce-order-prediction-user-analytics
The goal of this project is to build a machine learning model that predicts whether a user will place a new order within 7 days of their most recent purchase. This task addresses short-term user retention and helps identify high-risk churn users.

**YouTube Presentation Link:** https://youtu.be/r9bAPOIOeOM

## Dataset
Instacart Market Basket Analysis (Kaggle)
https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset/data
Download the public CSV files from Kaggle and store them in a version-controlled GitHub repository.

Data cleaning: handle missing values (e.g., days_since_prior_order for the first order), merge product metadata, ensure chronological ordering of user histories.
### Tables used
orders.csv – order metadata (user ID, order sequence, day-of-week, hour-of-day, days-since-prior-order)

order_products__prior.csv & order_products__train.csv – product IDs and quantities for each order

products.csv, aisles.csv, departments.csv – product and category details


## 1. Preliminary Visualizations

We performed exploratory analysis on the Instacart Online Grocery Dataset to understand user behavior patterns.

- **Basket size distribution** — Average ≈ **10 items**, with a heavy tail for bulk shoppers.
- **Order distribution by day of week** — Majority of orders occur on **weekends** (Sunday & Monday peaks).
- **Top departments purchased** — *Produce*, *Dairy & Eggs*, and *Snacks* dominate.
- **Days since prior order** — Right-skewed distribution; most users reorder within **0–10 days**, with a long-tail peak at 7 and 30.

## 2. Data Processing
For data processing, we started with files from the Instacart dataset — orders, order_products__prior, order_products__train, products, aisles, and departments.
We combined prior and train order tables to form a unified purchase history and filled missing values in days_since_prior_order with zeros, since the first order for each user naturally has no previous interval

Then, we merged product metadata (aisle and department) into the product table to get category-level information.
At the order level, we computed basket size (number of products per order) and appended it to each order record.

Next, we aggregated these into user-level features:

Average days between orders

Total number of orders per user

Average basket size

Number of unique departments purchased

Finally, we created a supervised label: for each order, label = 1 if the next order occurred within 7 days; otherwise 0.
This was implemented by shifting the order sequence within each user group to calculate the interval until the next purchase.

The final training table combines order-level and user-level features into a single dataframe ready for modeling.

## 3. Modeling Method
Our current baseline model is a Logistic Regression, chosen for its interpretability and as a benchmark for future models.

The feature matrix X includes both order-level and user-level variables:

days_since_prior_order, basket_size (short-term order behavior)

avg_days_between_orders, total_orders, avg_basket_size, and unique_departments (long-term user behavior).

We used an 80/20 stratified train-test split, keeping class balance.
The model was trained using scikit-learn’s LogisticRegression(max_iter=1000) and evaluated with multiple metrics: ROC-AUC, Average Precision (PR-AUC), Accuracy, Precision, Recall, and F1-score


## 3. Preliminary Results

Logistic regression achieved promising performance:

ROC-AUC: 0.815

Average Precision (PR-AUC): 0.745

Accuracy: 0.75

Precision: 0.69

Recall: 0.64

F1-score: 0.66

These results suggest that even simple linear modeling can capture key reordering behavior from user activity patterns.
Basket size, purchase frequency, and department diversity are strong indicators of short-term repurchase likelihood.
