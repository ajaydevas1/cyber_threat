import pandas as pd
from regex import T

TRAIN_PATH = "data/raw/KDDtrain+.txt"
TEST_PATH = "data/raw/KDDtest+.txt"

df_train = pd.read_csv(TRAIN_PATH, header=None)
df_test = pd.read_csv(TEST_PATH, header=None)

print("===== Raw Data =====")
print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

print("\n===== Sample Data of Train Data =====")
print(df_train.head(3))

print("\n===== Sample Data of Test Data =====")
print(df_test.tail(3))

