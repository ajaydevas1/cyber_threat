import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# -----------------------------
# STEP 1: Define fuzzy variables
# -----------------------------
ml_risk = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'ml_risk')
text_risk = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'text_risk')
seq_risk = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'seq_risk')

threat = ctrl.Consequent(np.arange(0, 101, 1), 'threat')

# -----------------------------
# STEP 2: Membership functions
# -----------------------------
for var in [ml_risk, text_risk, seq_risk]:
    var['low'] = fuzz.trimf(var.universe, [0, 0, 0.4])
    var['medium'] = fuzz.trimf(var.universe, [0.3, 0.5, 0.7])
    var['high'] = fuzz.trimf(var.universe, [0.6, 1, 1])

threat['low'] = fuzz.trimf(threat.universe, [0, 0, 40])
threat['medium'] = fuzz.trimf(threat.universe, [30, 50, 70])
threat['high'] = fuzz.trimf(threat.universe, [60, 100, 100])

# -----------------------------
# STEP 3: Fuzzy rules
# -----------------------------
rules = [
    ctrl.Rule(ml_risk['high'] | seq_risk['high'], threat['high']),
    ctrl.Rule(text_risk['high'], threat['high']),
    ctrl.Rule(ml_risk['medium'] & seq_risk['medium'], threat['medium']),
    ctrl.Rule(text_risk['medium'], threat['medium']),
    ctrl.Rule(ml_risk['low'] & text_risk['low'] & seq_risk['low'], threat['low'])
]

threat_ctrl = ctrl.ControlSystem(rules)
threat_sim = ctrl.ControlSystemSimulation(threat_ctrl)

# -----------------------------
# STEP 4: Load scores (example merge)
# -----------------------------
ml_df = pd.read_csv(r"C:\Users\Ajay\cyber_threat\data\processed\ml_combined_risk_score.csv")
text_df = pd.read_csv(r"C:\Users\Ajay\cyber_threat\data\processed\text_threat_scores.csv")
seq_df = pd.read_csv(r"C:\Users\Ajay\cyber_threat\data\processed\lstm_sequence_risk_scores.csv")

# Align length safely (demo purpose)
min_len = min(len(ml_df), len(text_df), len(seq_df))

results = []

for i in range(min_len):
    threat_sim.input['ml_risk'] = ml_df.loc[i, 'ml_risk_score']
    threat_sim.input['text_risk'] = text_df.loc[i, 'text_threat_score']
    threat_sim.input['seq_risk'] = seq_df.loc[i, 'sequence_risk_score']

    threat_sim.compute()
    results.append(threat_sim.output['threat'])

final_df = pd.DataFrame({
    "ml_risk": ml_df.loc[:min_len-1, 'ml_risk_score'],
    "text_risk": text_df.loc[:min_len-1, 'text_threat_score'],
    "seq_risk": seq_df.loc[:min_len-1, 'sequence_risk_score'],
    "final_threat_score": results
})

final_df.to_csv(
    "data/processed/final_threat_assessment.csv",
    index=False
)

print("=== FINAL THREAT DECISION SAMPLE ===")
print(final_df.head())
