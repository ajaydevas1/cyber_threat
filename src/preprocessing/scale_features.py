import pandas as pd
from sklearn.preprocessing import StandardScaler
from schema import NSL_KDD_COLUMNS

# Paths
TRAIN_PATH = "KDD_Train_Dataset"
TEST_PATH  = "KDD_Test_Dataset"

# -----------------------------
# Load & prepare data (same steps as before)
# -----------------------------
df_train = pd.read_csv(TRAIN_PATH, header=None)
df_test  = pd.read_csv(TEST_PATH, header=None)

df_train.columns = NSL_KDD_COLUMNS
df_test.columns  = NSL_KDD_COLUMNS

# Drop non-learning column
df_train.drop(columns=["difficulty_level"], inplace=True)
df_test.drop(columns=["difficulty_level"], inplace=True)

# Binary target
df_train["attack"] = df_train["label"].apply(lambda x: 0 if x == 0 else 1)
df_test["attack"]  = df_test["label"].apply(lambda x: 0 if x == 0 else 1)

df_train.drop(columns=["label"], inplace=True)
df_test.drop(columns=["label"], inplace=True)

# One-hot encoding
categorical_cols = ["protocol_type", "service", "flag"]
df_train = pd.get_dummies(df_train, columns=categorical_cols)
df_test  = pd.get_dummies(df_test, columns=categorical_cols)

# Align train & test
df_train, df_test = df_train.align(df_test, join="left", axis=1, fill_value=0)

# -----------------------------
# STEP 6: FEATURE SCALING
# -----------------------------
X_train = df_train.drop(columns=["attack"])
y_train = df_train["attack"]

X_test = df_test.drop(columns=["attack"])
y_test = df_test["attack"]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -----------------------------
# Save processed data
# -----------------------------
train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train_scaled_df["attack"] = y_train.values

test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_scaled_df["attack"] = y_test.values

train_scaled_df.to_csv("data/processed/nsl_kdd_train_scaled.csv", index=False)
test_scaled_df.to_csv("data/processed/nsl_kdd_test_scaled.csv", index=False)

print("=== SCALING COMPLETE ===")
print("Train shape:", train_scaled_df.shape)
print("Test shape:", test_scaled_df.shape)
