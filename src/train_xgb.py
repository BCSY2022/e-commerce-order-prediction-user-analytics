from pathlib import Path
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from load_data import load_raw_data, build_order_level_features
from build_dataset import add_label_within_7_days, assemble_training_table


def main(data_dir: str = "../data"):
    # ========= 1. Load raw data =========
    orders, order_products, products_full = load_raw_data(data_dir)

    # ========= 2. Construct features =========
    orders_enriched, user_features = build_order_level_features(
        orders, order_products, products_full
    )

    # ========= 3. Create 7-day label =========
    orders_labeled = add_label_within_7_days(orders_enriched)

    # ========= 4. Assemble dataset (X, y) =========
    df_model, X, y = assemble_training_table(orders_labeled, user_features)
    print(f"Total modeling rows: {len(df_model)}")

    # ========= 5. Train-test split =========
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ========= 6. XGBoost parameters =========
    model = XGBClassifier(
        n_estimators=300,          # boosting rounds
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",     # required to silence warning
        n_jobs=-1,
        random_state=42
    )

    print("\nTraining XGBoost...")
    model.fit(X_train, y_train)

    # ========= 7. Evaluate =========
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"\nXGBoost ROC-AUC: {roc:.4f}")
    print(f"XGBoost PR-AUC (Avg Precision): {pr_auc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # ========= 8. SAVE classification report =========
    Path("../figures").mkdir(exist_ok=True)

    report_text = classification_report(y_test, y_pred)
    with open("../figures/classification_report_xgboost.txt", "w") as f:
        f.write(report_text)

    # ========= 9. SAVE confusion matrix =========
    plt.figure(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title("Confusion Matrix (XGBoost)")
    plt.savefig("../figures/confusion_matrix_xgboost.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print("  ../figures/classification_report_xgboost.txt")
    print("  ../figures/confusion_matrix_xgboost.png")

    # predicted results
    df_test = df_model.loc[X_test.index].copy()  # include user_id, order_id, feature, label_within7days

    df_test["y_true"] = y_test
    df_test["y_pred"] = y_pred
    df_test["y_prob"] = y_prob

    out_dir = Path("../data")
    out_dir.mkdir(exist_ok=True)
    df_test.to_csv(out_dir / "predictions_logreg_xgb.csv", index=False)

if __name__ == "__main__":
    main()
