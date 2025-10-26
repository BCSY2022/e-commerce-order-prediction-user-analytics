import pandas as pd

def add_label_within_7_days(orders_enriched):
    """
    Give each (user_id, order_id) a supervised learning label:
    label = 1 if the next order occurs within <7 days
    label = 0 otherwise
    Rules:
    - For the same user, sort by order_number
    - days_since_prior_order represents the interval between the current order and the previous order
    - We want to know how many days separate the "next order" from the "current order."
    This can be done by shifting (-1) to get the days_since_prior_order of the next order within the user group.
    """

    orders_sorted = (
        orders_enriched
        .sort_values(["user_id", "order_number"])
        .copy()
    )

    # The time difference between the next order and the "current order":
    # use the days_since_prior_order in the next row
    orders_sorted["days_until_next_order"] = (
        orders_sorted
        .groupby("user_id")["days_since_prior_order"]
        .shift(-1)
    )

    # < 7 days = 1, otherwise = 0
    # Note: The last order has no next order -> NaN -> treated as 0
    orders_sorted["label_within7days"] = (
        (orders_sorted["days_until_next_order"] < 7)
        .fillna(False)
        .astype(int)
    )

    return orders_sorted


def assemble_training_table(orders_labeled, user_features):
    """
    Combine order-level rows and user-level features to form a table
    that can be used directly for training.
    """

    df = orders_labeled.merge(user_features, on="user_id", how="left")

    # feature columns
    feature_cols = [
        "days_since_prior_order",     # The number of days between the current order and the last order
        "basket_size",                # Current shopping basket size
        "avg_days_between_orders",    # Average order interval for users
        "total_orders",               # Cumulative number of orders placed by users
        "avg_basket_size",            # Average shopping basket size of users
        "unique_departments",         # The number of different departments the user has purchased
    ]

    # Target column
    target_col = "label_within7days"

    # Drop rows with missing values to ensure the model can be trained directly
    df_model = df.dropna(subset=feature_cols + [target_col]).copy()

    X = df_model[feature_cols]
    y = df_model[target_col]

    return df_model, X, y
