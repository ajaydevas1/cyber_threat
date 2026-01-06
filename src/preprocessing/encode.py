import pandas as pd
from schema import NSL_KDD_COLUMNS

# Paths
TRAIN_PATH = "data/raw/KDDTrain+.txt"
TEST_PATH  = "data/raw/KDDTest+.txt"

# Load raw data
df_train = pd.read_csv(TRAIN_PATH, header=None)
df_test  = pd.read_csv(TEST_PATH, header=None)

# Apply schema
df_train.columns = NSL_KDD_COLUMNS
df_test.columns  = NSL_KDD_COLUMNS

# Drop non-learning column
df_train.drop(columns=["difficulty_level"], inplace=True)
df_test.drop(columns=["difficulty_level"], inplace=True)

# Create binary target
df_train["attack"] = df_train["label"].apply(lambda x: 0 if x == 0 else 1)
df_test["attack"]  = df_test["label"].apply(lambda x: 0 if x == 0 else 1)

df_train.drop(columns=["label"], inplace=True)
df_test.drop(columns=["label"], inplace=True)

# -----------------------------
# STEP 5: One-hot encode categorical columns
# -----------------------------
categorical_cols = ["protocol_type", "service", "flag"]

df_train_encoded = pd.get_dummies(df_train, columns=categorical_cols)
df_test_encoded  = pd.get_dummies(df_test, columns=categorical_cols)

# Align train and test (CRITICAL)
df_train_encoded, df_test_encoded = df_train_encoded.align(
    df_test_encoded, join="left", axis=1, fill_value=0
)

# -----------------------------
# VERIFICATION
# -----------------------------
print("=== CATEGORICAL ENCODING DONE ===")
print("Train shape:", df_train_encoded.shape)
print("Test shape:", df_test_encoded.shape)

print("\nSample encoded columns:")
print([col for col in df_train_encoded.columns if col.startswith("protocol_type_")][:5])

print("\nSample rows:")
print(df_train_encoded.head(3))
