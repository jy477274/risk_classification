import os
import tempfile
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import requests
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.frozen import FrozenEstimator


#Point to local MLFLow Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("trust_safety_risk_classifier")

#Load Real Public Fraud Dataset (284,807 txns, ~0.172% fraud rate)
os.makedirs("data", exist_ok=True)
DATA_PATH = os.path.join("data", "creditcard.csv")
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

if not os.path.exists(DATA_PATH):
    print(f"Downloading creditcard.csv (~144 MB) from Google Cloud mirror...")
    with requests.get(DATA_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(DATA_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\rProgress: {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({downloaded / total:.0%})", end="")
    print("\nDownload complete!")
else:
    print(f"Loading cached dataset from {DATA_PATH}...")

df = pd.read_csv(DATA_PATH)
X_df = df.drop(columns=["Class"]).astype(np.float64)
y = df["Class"].astype(int).to_numpy()

print(f"Loaded {len(X_df):,} rows | Positive fraud rate: {y.mean():.4%}")

#Split 60% Train, 20% Calibration, 20% Test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_df, y, test_size=0.20, stratify=y, random_state=42
)

X_train, X_cal, y_train, y_cal = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

#Simulate fixed human-review queue capacity (top k% of volume)
def compute_precision_at_k(y_true, y_prob, k_frac=0.01):
    #max num of items reviewed in a period
    k = max(1, int(len(y_true) * k_frac))
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    precision_at_k = float(np.mean(y_true[top_k_idx]))
    recall_at_k = float(np.sum(y_true[top_k_idx]) / np.sum(y_true))
    cutoff_thresh = float(y_prob[top_k_idx[-1]])
    return precision_at_k, recall_at_k, cutoff_thresh


#Recall at 1% FPR: Max vio capture while capping false alarms <= 1%
def compute_recall_at_max_fpr(y_true, y_prob, max_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = np.where(fpr <= max_fpr)[0]
    best_idx = valid[-1]
    return float(tpr[best_idx]), float(thresholds[best_idx])

def compute_brier_skill_score(y_true, y_prob):
    base_rate = np.full_like(y_true, fill_value=np.mean(y_true), dtype=float)
    brier_ref = brier_score_loss(y_true, base_rate)
    brier_model = brier_score_loss(y_true, y_prob)
    return float(1.0 - (brier_model / brier_ref))

#find max recall achievable while guaranteeing precision >= min_precision
def compute_recall_at_precision(y_true, y_prob, min_precision=0.90):
   precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
   valid = np.where(precisions[:-1] >= min_precision)[0]

   if len(valid) == 0:
      return 0.0, 1.0

   best_index = valid[np.argmax(recalls[valid])]
   return float(recalls[best_index]), float(thresholds[best_index])


# ---------------------------------------------------------
# Rich Visual Artifacts for MLflow
# ---------------------------------------------------------
def plot_calibration_comparison(y_true, raw_prob, cal_prob, method_name, save_path):
    """2-panel Reliability Diagram: Uniform Bins (tail calibration) vs. Quantile Bins."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, strategy, title in [
        (axes[0], "uniform", "Uniform Bins (Exposes High-Risk Tail Distortion)"),
        (axes[1], "quantile", "Quantile Bins (Equal Sample Density)"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect Calibration")
        for label, probs, color in [
            ("Raw LightGBM (Weighted)", raw_prob, "#1f77b4"),
            (f"Calibrated ({method_name})", cal_prob, "#ff7f0e"),
        ]:
            frac_pos, mean_pred = calibration_curve(
                y_true, probs, n_bins=10, strategy=strategy
            )
            ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, color=color, label=label)

        ax.set_xlabel("Mean Predicted Risk Probability")
        ax.set_ylabel("Actual Fraud Fraction")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_operating_curves(y_true, cal_prob, thresh_fpr, thresh_top_k, save_path):
    """2-panel PR & ROC curves highlighting actual business operating thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Precision-Recall Curve
    prec, rec, pr_thresh = precision_recall_curve(y_true, cal_prob)
    ap = average_precision_score(y_true, cal_prob)
    axes[0].plot(rec, prec, color="#2ca02c", linewidth=2, label=f"PR Curve (AP = {ap:.3f})")

    # Mark Top-1% Queue operating point
    idx_k = np.argmin(np.abs(pr_thresh - thresh_top_k))
    axes[0].scatter(
        rec[idx_k], prec[idx_k], color="red", zorder=5, s=70,
        label=f"Top 1% Queue (P={prec[idx_k]:.1%}, R={rec[idx_k]:.1%})"
    )
    axes[0].set_xlabel("Recall (Fraud Capture Rate)")
    axes[0].set_ylabel("Precision (Review Queue Purity)")
    axes[0].set_title("Precision-Recall Curve & Queue Operating Point")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower left")

    # Panel 2: ROC Curve (Zoomed to 0-10% FPR where risk engines operate)
    fpr, tpr, roc_thresh = roc_curve(y_true, cal_prob)
    auc = roc_auc_score(y_true, cal_prob)
    axes[1].plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"ROC Curve (AUC = {auc:.4f})")
    axes[1].axvline(0.01, color="red", linestyle="--", alpha=0.7, label="1% FPR Ceiling")
    axes[1].set_xlim([0.0, 0.10])
    axes[1].set_ylim([0.0, 1.02])
    axes[1].set_xlabel("False Positive Rate (Zoomed 0% - 10%)")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].set_title("Low-FPR Operating Region")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(lgbm_model, feature_names, save_path, top_n=15):
    """Horizontal bar chart of top split/gain features."""
    importances = lgbm_model.booster_.feature_importance(importance_type="gain")
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_names)[idx], importances[idx], color="#4c72b0")
    ax.set_xlabel("Total Gain Importance")
    ax.set_title(f"Top {top_n} Risk Signals (LightGBM Gain)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def build_threshold_table(y_true, cal_prob):
    """Generates a CSV lookup table for PMs/Ops showing volume vs. precision/recall trade-offs."""
    rows = []
    for thresh in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]:
        preds = (cal_prob >= thresh).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn)
        flag_rate = (tp + fp) / len(y_true)
        rows.append({
            "threshold": thresh,
            "flagged_volume_pct": round(flag_rate * 100, 3),
            "precision_pct": round(precision * 100, 2),
            "recall_pct": round(recall * 100, 2),
            "fpr_pct": round(fpr * 100, 3),
            "true_positives": tp,
            "false_positives": fp,
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------
# 3. Train, Calibrate, Log Rich Artifacts & Promote Model
# ---------------------------------------------------------
#params = {
#    "n_estimators": 250,
#    "learning_rate": 0.05,
#    "num_leaves": 31,
#    "scale_pos_weight": 10.0, #intentional weight imbalance distorts raw probs to show calibration use
#    "random_state": 42,
#}
calibration_method = "sigmoid"

#params side-by-side test
params = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

with mlflow.start_run(run_name="lgbm_creditcard_platt_calibrated") as run:
    mlflow.log_params(params)
    mlflow.log_param("calibration_method", calibration_method),
    mlflow.log_param("dataset", "openml_creditcard_v1")
    mlflow.log_param("positive_rate_pct", round(float(y.mean() * 100), 4))

    #train base LightGBM
    base_model = LGBMClassifier(**params, verbose = -1)
    base_model.fit(X_train, y_train)

    #calibrate on the held out calibration set
    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model), method=calibration_method
    )
    calibrated_model.fit(X_cal, y_cal)

    #evaluate base and calibrated models on the test set
    raw_probs = base_model.predict_proba(X_test)[:, 1]
    cal_probs = calibrated_model.predict_proba(X_test)[:, 1]

    #compute custom risk metrics
    prec_at_1pct, rec_at_1pct, thresh_1pct = compute_precision_at_k(
        y_test, cal_probs, k_frac=0.01
    )

    rec_1pct_fpr, thresh_1pct_fpr = compute_recall_at_max_fpr(
        y_test, cal_probs, max_fpr=0.01
    )

    bss_raw = compute_brier_skill_score(y_test, raw_probs)
    bss_cal = compute_brier_skill_score(y_test, cal_probs)

    #log standard and custom business risk metrics in mlflow
    mlflow.log_metrics({
        "roc_auc": roc_auc_score(y_test, cal_probs),
        "pr_auc": average_precision_score(y_test, cal_probs),
        "brier_score_raw": brier_score_loss(y_test, raw_probs),
        "brier_score_calibrated": brier_score_loss(y_test, cal_probs),
        "precision_at_top_1pct_queue": prec_at_1pct,
        "recall_at_top_1pct_queue": rec_at_1pct,
        "recall_at_1pct_fpr": rec_1pct_fpr,
        "threshold_at_1pct_fpr": thresh_1pct_fpr,
        "brier_skill_score_raw": bss_raw,
        "brier_skill_score_calibrated": bss_cal,
    })

    # Generate and log artifacts in MLFlow
    with tempfile.TemporaryDirectory() as tempdir:
        cal_plot = os.path.join(tempdir, "calibration_curve.png")
        ops_plot = os.path.join(tempdir, "pr_roc_operating_points.png")
        imp_plot = os.path.join(tempdir, "feature_importance.png")
        thresh_csv = os.path.join(tempdir, "threshold_operating_points.csv")
        fp_csv = os.path.join(tempdir, "top_false_positives.csv")

        plot_calibration_comparison(y_test, raw_probs, cal_probs, calibration_method, cal_plot)
        plot_operating_curves(y_test, cal_probs, thresh_1pct_fpr, thresh_1pct, ops_plot)
        plot_feature_importance(base_model, X_df.columns.tolist(), imp_plot)

        #save threshold tradeoff table
        thresh_df = build_threshold_table(y_test, cal_probs)
        thresh_df.to_csv(thresh_csv, index=False)

        #save top 25 false positives for analysis
        audit_df = X_test.copy()
        audit_df["actual_label"] = y_test
        audit_df["calibrated_risk_prob"] = cal_probs
        top_false_pos = (
            audit_df[audit_df["actual_label"] == 0].sort_values("calibrated_risk_prob", ascending=False).head(25)
        )
        top_false_pos.to_csv(fp_csv, index=False)

        #log artifacts
        mlflow.log_artifact(cal_plot, artifact_path="plots")
        mlflow.log_artifact(ops_plot, artifact_path="plots")
        mlflow.log_artifact(imp_plot, artifact_path="plots")
        mlflow.log_artifact(thresh_csv, artifact_path="reports")
        mlflow.log_artifact(fp_csv, artifact_path="reports")

    # Log & Register the calibrated pipeline with an input_example (infers schema!)
    model_info = mlflow.sklearn.log_model(
        sk_model=calibrated_model,
        name="risk_classifier",
        input_example=X_test.iloc[:3],
        registered_model_name="trust-safety-classifier",
        skops_trusted_types=[
            "collections.OrderedDict",
            "lightgbm.basic.Booster",
            "lightgbm.sklearn.LGBMClassifier",
            "sklearn.calibration._CalibratedClassifier",
            "sklearn.calibration._SigmoidCalibration",
            "sklearn.frozen.FrozenEstimator",

        ],
    )

    # Assign '@champion' alias in Model Registry
    client = MlflowClient()
    new_version = model_info.registered_model_version
    client.set_registered_model_alias(
        name="trust-safety-classifier",
        alias="champion",
        version=str(new_version),
    )
    print(f"\nSuccess! Run ID: {run.info.run_id}")
    print(f"Registered Model Version {new_version} promoted to @champion")
    print(f"PR-AUC: {average_precision_score(y_test, cal_probs):.4f} | Recall@1%FPR: {rec_1pct_fpr:.2%}")
    print(f"Brier Skill Score improved: {bss_raw:.4f} (raw) -> {bss_cal:.4f} (calibrated)")