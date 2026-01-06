import numpy as np
import pandas as pd
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\Ajay\cyber_threat\data\processed\nsl_kdd_train_scaled.csv')

X = df.drop(columns=['attack']).values
y = df['attack'].values

SEQ_Len = 10

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len]) 
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y, SEQ_Len)

print("Seqence Shape:", X_seq.shape)

X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(32, input_shape=(SEQ_Len, X.shape[1])),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val),
    epochs=5,
    batch_size=64
)

sequence_risk_scores = model.predict(X_val).flatten()

seq_df = pd.DataFrame({
    'sequence_risk_score': sequence_risk_scores,
    'true_label': y_val
})

seq_df.to_csv('data/processed/lstm_sequence_risk_scores.csv', index=False)

print("=== SEQUENCE RISK SAMPLE ===")
print(seq_df.head())
print("\nScore statistics:")
print(seq_df["sequence_risk_score"].describe())