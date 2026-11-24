import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

#Connect to MLFlow and load the champion model
mlflow.set_tracking_uri("http://127.0.0.1:5000")
MODEL_URI = "models:/trust-safety-classifier@champion"
print("Loading model from {MODEL_URI}...")
champion_model = mlflow.sklearn.load_model(MODEL_URI)

#load same test split from the credit card data
df = pd.read_csv("data/creditcard.csv")
X_df = df.drop(columns=["Class"]).astype(np.float64)
y = df["Class"].astype(int).to_numpy()

_, X_test, _, y_test = train_test_split(
    X_df, y, test_size=0.20, stratify=y, random_state=42
)

#generate predictions from champion model and keep those above the 1% fpr thresh
probs = champion_model.predict_proba(X_test)[:, 1]
THRESHOLD = 0.0057
preds = (probs >= THRESHOLD).astype(int)

#define business critical data slices with which to eval on
eval_df = X_test.copy()
eval_df["y_true"] = y_test
eval_df["y_prob"] = probs
eval_df["y_pred"] = preds

slices = {
    "All Traffic (Baseline)": eval_df,
    "Micro Transactions": eval_df[eval_df["Amount"] < 10.0],
    "Mid-Tier Transactions": eval_df[(eval_df["Amount"] >= 10.0) & (eval_df["Amount"] <= 200.0)],
    "High-Value Transactions": eval_df[eval_df["Amount"] > 200.0],
    "Window Day 1": eval_df[eval_df["Time"] <= 86400],
    "Window Day 2": eval_df[eval_df["Time"] > 86400]
}

#compute per-slice metrics and check against 1.5% fpr guardrail
print("\nSlice Evaluation Report:\n")
rows = []
for slice_name, sub in slices.items():
    yt = sub["y_true"].to_numpy()
    yp = sub["y_pred"].to_numpy()
    yprob = sub["y_prob"].to_numpy()

    fp = int(np.sum((yp == 1) & (yt == 0)))
    tn = int(np.sum((yp == 0) & (yt == 0)))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    rows.append({
        "Slice": slice_name,
        "Count": len(sub),
        "Fraud Cases": int(yt.sum()),
        "ROC AUC": round(roc_auc_score(yt, yprob), 4),
        "PR AUC": round(average_precision_score(yt, yprob), 4),
        "Recall %": round(recall_score(yt, yp, zero_division=0) * 100, 2),
        "Precision %": round(precision_score(yt, yp, zero_division=0) * 100, 2),
        "False Positive Rate": round(fpr * 100, 3),
        "Guardrail Status": "ALERT (>1.5% FPR)" if fpr > 0.015 else "PASS",
    })

report_df = pd.DataFrame(rows)
print(report_df.to_string(index=False))