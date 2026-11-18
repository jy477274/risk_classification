import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.frozen import FrozenEstimator


#Point to local MLFLow Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("trust_safety_risk_classifier")

#Generate synthetic imbalanced data
X, y = make_classification(
    n_samples = 20000,
    n_features = 12,
    n_informative = 8,
    weights = [0.985, 0.015],
    random_state = 42
)

feature_names = [f"risk_signal_{i}" for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)

#Split 60% Train, 20% Calibration, 20% Test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_df, y, test_size=0.20, stratify=y, random_state=42
)

X_train, X_cal, y_train, y_cal = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

#find max recall achievable while guaranteeing precision >= min_precision
def compute_recall_at_precision(y_true, y_prob, min_precision=0.90):
   precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
   valid = np.where(precisions[:-1] >= min_precision)[0]

   if len(valid) == 0:
      return 0.0, 1.0

   best_index = valid[np.argmax(recalls[valid])]
   return float(recalls[best_index]), float(thresholds[best_index])


#generate side-by-side Reliability Diagram showing before and after calibration
def plot_calibration(y_true, raw_prob, cal_prob, filename = "calibration_curve.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    #isotonic cal instead of platt here
    for name, probs in [("Raw LightGBM", raw_prob), ("Isotonic Calibrated", cal_prob)]:
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)

    ax.set_xlabel("Mean Predicted Risk Probability")
    ax.set_ylabel("Actual Violation Fraction")
    ax.set_title("Reliability Diagram (Quantile Bins)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return filename

#run and log experiment in mlflow
params = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "scale_pos_weight": 10.0, #intentional weight imbalance distorts raw probs to show calibration use
    "random_state": 42,

}

with mlflow.start_run(run_name="lgbm_isotonic_calibration") as run:
    mlflow.log_params(params)
    mlflow.log_param("calibration_method", "isotonic")

    #train base LightGBM
    base_model = LGBMClassifier(**params, verbose = -1)
    base_model.fit(X_train, y_train)

    #calibrate on the held out calibration set
    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model), method="isotonic"
    )
    calibrated_model.fit(X_cal, y_cal)

    #evaluate base and calibrated models on the test set
    raw_probs = base_model.predict_proba(X_test)[:, 1]
    cal_probs = calibrated_model.predict_proba(X_test)[:, 1]

    recall_at_90p, opt_thresh = compute_recall_at_precision(
        y_test, cal_probs, min_precision=0.90
    )

    #log standard and custom business risk metrics in mlflow
    mlflow.log_metrics({
        "roc_auc": roc_auc_score(y_test, cal_probs),
        "pr_auc": average_precision_score(y_test, cal_probs),
        "brier_score_raw": brier_score_loss(y_test, raw_probs),
        "brier_score_calibrated": brier_score_loss(y_test, cal_probs),
        "recall_at_90p_prec": recall_at_90p,
        "optimal_threshold_90p_prec": opt_thresh,
    })

    # Log Calibration Curve image as an artifact
    plot_path = plot_calibration(y_test, raw_probs, cal_probs)
    mlflow.log_artifact(plot_path, artifact_path="plots")

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
            "sklearn.frozen.FrozenEstimator",
        ],
    )

    # Assign '@champion' alias in Model Registry
    client = MlflowClient()
    client.set_registered_model_alias(
        name="trust-safety-classifier",
        alias="champion",
        version="1",
    )
    print(f"Logged Run ID: {run.info.run_id}")