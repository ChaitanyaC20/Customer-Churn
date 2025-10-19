import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, RocCurveDisplay
from app.config import MODEL_FILE, FEATURES_FILE, METRICS_FILE, ARTIFACT_DIR


def load_saved_model():
    """Load trained model from saved .joblib file"""
    print(f"📦 Loading model from {MODEL_FILE}")
    return joblib.load(MODEL_FILE)

def load_metrics():
    """Load saved metrics from JSON file"""
    try:
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)
        print(f"📊 Loaded metrics: {metrics}")
        return metrics
    except Exception as e:
        print(f"⚠️ Could not load metrics: {e}")
        return {}

def plot_feature_importance(model, feature_columns, top_n=15):
    """Plot top N feature importances for XGBoost"""
    try:
        xgb = model.named_steps['xg']
        importances = xgb.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 5))
        sns.barplot(data=feat_df, x='Importance', y='Feature', palette='Blues_d')
        plt.title(f"Top {top_n} Feature Importances")
        plt.tight_layout()
        out_path = os.path.join(ARTIFACT_DIR, "feature_importance.png")
        plt.savefig(out_path)
        plt.close()
        print(f"✅ Saved feature importance plot → {out_path}")
        return out_path
    except Exception as e:
        print(f"⚠️ Could not plot feature importance: {e}")
        return None

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    out_path = os.path.join(ARTIFACT_DIR, "confusion_matrix_eval.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved confusion matrix → {out_path}")
    return out_path

def plot_roc_curve(y_true, y_probs):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.tight_layout()
    out_path = os.path.join(ARTIFACT_DIR, "roc_curve.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved ROC curve → {out_path}")
    return out_path

def evaluate_model(X_test, y_test, feature_columns):
    """Evaluate model with feature importance, confusion matrix, ROC, and metrics"""
    model = load_saved_model()
    metrics = load_metrics()

    # Predict
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.3).astype(int)

    # Reports
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Plots
    fi_path = plot_feature_importance(model, feature_columns)
    cm_path = plot_confusion_matrix(y_test, y_pred)
    roc_path = plot_roc_curve(y_test, y_probs)

    print("\n✅ Evaluation complete. Outputs saved:")
    print(f"- Feature Importance → {fi_path}")
    print(f"- Confusion Matrix → {cm_path}")
    print(f"- ROC Curve → {roc_path}")
    print(f"- Metrics → {METRICS_FILE}")

if __name__ == "__main__":
    from app.data import load_raw_data
    from app.preprocess import preprocess_customer
    from sklearn.model_selection import train_test_split

    print("📥 Loading data...")
    df = load_raw_data()
    X, y, feature_columns = preprocess_customer(df)
    _, X_test, _, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=44)
    print("✅ Data ready for evaluation")

    evaluate_model(X_test, y_test, feature_columns)