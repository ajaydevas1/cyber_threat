import pandas as pd

# Load existing ML scores
knn_df = pd.read_csv(r"C:\Users\Ajay\cyber_threat\data\processed\knn_anomaly_scores.csv")
svm_df = pd.read_csv(r"C:\Users\Ajay\cyber_threat\data\processed\svm_risk_scores.csv")

# Basic sanity check
if len(knn_df) != len(svm_df):
    raise ValueError("KNN and SVM score files length mismatch")

# Combine scores (equal weight)
alpha = 0.5

combined_df = pd.DataFrame({
    "knn_score": knn_df["anomaly_score"],
    "svm_score": svm_df["svm_risk_score"],
    "ml_risk_score": alpha * knn_df["anomaly_score"] + (1 - alpha) * svm_df["svm_risk_score"],
    "true_label": knn_df["true_label"]
})

# Save combined ML score
combined_df.to_csv(
    "data/processed/ml_combined_risk_score.csv",
    index=False
)

print("✅ ML combined risk score created")
print(combined_df.head())
