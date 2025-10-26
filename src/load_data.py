import pandas as pd
import numpy as np
from pathlib import Path

def load_raw_data(data_dir="../data"):
    data_dir = Path(data_dir)

    orders = pd.read_csv(data_dir / "orders.csv")
    prior = pd.read_csv(data_dir / "order_products__prior.csv")
    train = pd.read_csv(data_dir / "order_products__train.csv")
    products = pd.read_csv(data_dir / "products.csv")
    aisles = pd.read_csv(data_dir / "aisles.csv")
    departments = pd.read_csv(data_dir / "departments.csv")

    # combine prior + train
    order_products = pd.concat([prior, train], ignore_index=True)

    # Missing value processing: The first order has no previous order, so it is NaN (Filling 0)
    orders["days_since_prior_order"] = orders["days_since_prior_order"].fillna(0)

    # Add aisle/department names to the product table
    products_full = (
        products
        .merge(aisles, on="aisle_id", how="left")
        .merge(departments, on="department_id", how="left")
    )

    return orders, order_products, products_full


def build_order_level_features(orders, order_products, products_full):
    """
    Returns two items:
    - orders_enriched: Each order with information such as basket_size
    - user_features: Aggregated features for each user
    """

    # Calculate the shopping basket size for each order
    basket_size_per_order = (
        order_products
        .groupby("order_id")["product_id"]
        .count()
        .rename("basket_size")
        .reset_index()
    )

    # Add basket_size to the order
    orders_enriched = orders.merge(basket_size_per_order,
                                   on="order_id",
                                   how="left")

    # Combine order details with product classification information
    order_products_enriched = (
        order_products
        .merge(orders[["order_id", "user_id"]], on="order_id", how="left")
        .merge(products_full[["product_id", "department", "aisle"]],
               on="product_id", how="left")
    )

    # Count how many different departments each user has purchased
    user_department_diversity = (
        order_products_enriched
        .groupby("user_id")["department"]
        .nunique()
        .rename("unique_departments")
        .reset_index()
    )

    # Make more aggregate features for each user
    user_features = (
        orders_enriched
        .groupby("user_id")
        .agg(
            avg_days_between_orders=("days_since_prior_order", "mean"),
            total_orders=("order_number", "max"),
            avg_basket_size=("basket_size", "mean"),
        )
        .reset_index()
    )

    # Merge unique_departments
    user_features = user_features.merge(
        user_department_diversity,
        on="user_id",
        how="left"
    )

    return orders_enriched, user_features
