from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)

from load_data import load_raw_data, build_order_level_features
from build_dataset import add_label_within_7_days, assemble_training_table
from pathlib import Path

def main(data_dir: str = "../data"):

    orders, order_products, products_full = load_raw_data(data_dir)

    # build features
    orders_enriched, user_features = build_order_level_features(
        orders, order_products, products_full
    )

    # add labels
    orders_labeled = add_label_within_7_days(orders_enriched)

    # make training (X, y)
    df_model, X, y = assemble_training_table(orders_labeled, user_features)

    print(f"Total rows for modeling: {len(df_model)}")

    # train/ test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Random Forest model
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=20,
        min_samples_split=50,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    # train
    print("Training Random Forest...")
    rf.fit(X_train, y_train)

    # predict and evaluate
    y_prob = rf.predict_proba(X_test)[:, 1]   # probability
    y_pred = rf.predict(X_test)              # classfication label，calculate precision/recall

    roc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    report_text = classification_report(y_test, y_pred)

    print(f"\nRandom Forest ROC-AUC: {roc:.4f}")
    print(f"Random Forest PR-AUC (Avg Precision): {pr_auc:.4f}\n")
    print("\nClassification Report:\n", report_text)
    output_dir = Path("../figures")
    output_dir.mkdir(exist_ok=True)

    # Save into a text file
    with open(output_dir / "classification_report_RF.txt", "w") as f:
        f.write(report_text)

if __name__ == "__main__":
    main()
