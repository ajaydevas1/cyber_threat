import pandas as pd
from sklearn.svm import SVC

# Load scaled data
train_df = pd.read_csv("data/processed/nsl_kdd_train_scaled.csv")
test_df  = pd.read_csv("data/processed/nsl_kdd_test_scaled.csv")

# Split features and target
X_train = train_df.drop(columns=["attack"])
y_train = train_df["attack"]

X_test = test_df.drop(columns=["attack"])
y_test = test_df["attack"]

# -----------------------------
# Train SVM
# -----------------------------
svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True
)

print("Training SVM (this may take some time)...")
svm.fit(X_train, y_train)


svm_attack_prob = svm.predict_proba(X_test)[:, 1]

# Create dataframe
svm_score_df = pd.DataFrame({
    "svm_risk_score": svm_attack_prob,
    "true_label": y_test
})


svm_score_df.to_csv(
    "data/processed/svm_risk_scores.csv",
    index=False
)

print("=== SVM RISK SCORE SAMPLE ===")
print(svm_score_df.head())

print("\nSVM risk score statistics:")
print(svm_score_df["svm_risk_score"].describe())
