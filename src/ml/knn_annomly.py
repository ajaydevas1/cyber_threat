import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

train_df = pd.read_csv(r'processed_nsl_kdd_train_scaled_dataset') # Obtain the processed dataset by running python code 
test_df = pd.read_csv(r'processed_nsl_kdd_test_scaled_dataset') # Obtain the processed dataset by running python code 

X_train = train_df.drop(columns=['attack'])
y_train = train_df['attack']

X_test = test_df.drop(columns=['attack'])
y_test = test_df['attack']

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

attack_prob = knn.predict_proba(X_test)[:, 1]

anomaly_df = X_test.copy()
anomaly_df["true_label"] = y_test.values
anomaly_df["anomaly_score"] = attack_prob


anomaly_df[["anomaly_score", "true_label"]].to_csv(
    "data/processed/knn_anomaly_scores.csv",
    index=False
)

# Quick inspection
print("=== ANOMALY SCORE SAMPLE ===")
print(anomaly_df[["anomaly_score", "true_label"]].head())
print("\nAnomaly score statistics:")
print(anomaly_df["anomaly_score"].describe())
