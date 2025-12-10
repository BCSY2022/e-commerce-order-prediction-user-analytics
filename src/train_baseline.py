from pathlib import Path
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, average_precision_score

from load_data import load_raw_data, build_order_level_features
from build_dataset import add_label_within_7_days, assemble_training_table


def main():
    # 1. Read raw data + basic cleaning + feature aggregation
    orders, order_products, products_full = load_raw_data(data_dir="../data")
    orders_enriched, user_features = build_order_level_features(
        orders, order_products, products_full
    )

    # 2. Tag (whether there is next order within 7 days)
    orders_labeled = add_label_within_7_days(orders_enriched)

    # 3. Set up a training table
    df_model, X, y = assemble_training_table(orders_labeled, user_features)

    # ===== Correlation matrix visualization =====
    # Combine X and y into a single DataFrame so that label correlation is included
    corr_df = pd.concat([X, y.rename("label_within_7days")], axis=1)
    # Compute correlations
    corr = corr_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Matrix of Features and Target")

    # Make sure output folder exists
    Path("../figures").mkdir(exist_ok=True)

    # SAVE IMAGE
    plt.savefig("../figures/feature_corr.png", dpi=200, bbox_inches="tight")

    plt.close()  # close figure to avoid memory warnings
    print("Correlation matrix saved to: ../figures/feature_corr.png")
    # Compute correlations
    corr = corr_df.corr()


    # 4. Simple split train/test
    # random partitioning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. train baseline (Logistic Regression)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    report_text = classification_report(y_test, y_pred)
    print("ROC-AUC:", roc_auc)
    print("Average Precision (PR-AUC-ish):", avg_precision)
    print("\nClassification Report:\n", report_text)

    output_dir = Path("../figures")
    output_dir.mkdir(exist_ok=True)

    # Save into a text file
    with open(output_dir / "classification_report_logistic.txt", "w") as f:
        f.write(report_text)

if __name__ == "__main__":
    main()
