import pandas as pd
from schema import NSL_KDD_COLUMNS

TRAIN_PATH = "data/raw/KDDTrain+.txt"
TEST_PATH  = "data/raw/KDDTest+.txt"

# Load raw data
df_train = pd.read_csv(TRAIN_PATH, header=None)
df_test  = pd.read_csv(TEST_PATH, header=None)

# Apply schema
df_train.columns = NSL_KDD_COLUMNS
df_test.columns  = NSL_KDD_COLUMNS

# --- VERIFICATION ---
print("=== SCHEMA APPLIED ===")
print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

print("\nLast 5 column names:")
print(df_train.columns[-5:])

print("\nSample rows (selected columns):")
print(df_train[["duration", "protocol_type", "service", "src_bytes", "label"]].head())