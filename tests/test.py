"""
Simple sanity tests for the Instacart order prediction pipeline.
"""

import pandas as pd

from src.build_dataset import add_label_within_7_days, assemble_training_table


def test_add_label_within_7_days_adds_binary_column():
    """Check that add_label_within_7_days adds a 0/1 label column."""

    data = {
        "user_id": [1, 1, 1, 2, 2],
        "order_id": [10, 11, 12, 20, 21],
        "order_number": [1, 2, 3, 1, 2],
        "days_since_prior_order": [None, 3.0, 5.0, None, 10.0],
    }
    orders_enriched = pd.DataFrame(data)

    out = add_label_within_7_days(orders_enriched)

    # should be another label col
    assert "label_within7days" in out.columns, "label_within7days column missing"

    # label can be 0 or 1
    values = set(out["label_within7days"].dropna().unique())
    assert values.issubset({0, 1}), f"Unexpected label values: {values}"


def test_assemble_training_table_shapes():
    """Check that assemble_training_table returns consistent shapes."""

    orders_labeled = pd.DataFrame({
        "user_id": [1, 1, 2],
        "order_id": [10, 11, 20],
        "order_number": [1, 2, 1],
        "days_since_prior_order": [None, 3.0, None],
        "label_within7days": [1, 0, 1],
        "order_dow": [0, 1, 2],
        "order_hour_of_day": [10, 15, 20],
        "basket_size": [5, 8, 3],
    })

    user_features = pd.DataFrame({
        "user_id": [1, 2],
        "avg_days_between_orders": [3.0, 10.0],
        "total_orders": [2, 1],
        "avg_basket_size": [6.5, 3.0],
        "unique_departments": [4, 2],
    })

    df_model, X, y = assemble_training_table(orders_labeled, user_features)

    # should be the same rows
    assert len(df_model) == len(X) == len(y), "df_model / X / y row counts mismatch"

    # y can only be 0 or 1
    values = set(y.unique())
    assert values.issubset({0, 1}), f"Unexpected target values: {values}"


def main():
    print("Running sanity tests...")
    test_add_label_within_7_days_adds_binary_column()
    print("add_label_within_7_days OK")
    test_assemble_training_table_shapes()
    print("assemble_training_table OK")
    print("All sanity tests passed.")


if __name__ == "__main__":
    main()
