import streamlit as st
import pandas as pd

st.title("Hybrid Cyber Threat Detection System")

st.markdown("### Model Outputs")

# Load precomputed results
final_df = pd.read_csv(r"C:\Users\Ajay\cyber_threat\data\processed\final_threat_assessment.csv")

index = st.slider(
    "Select sample index",
    0, len(final_df)-1, 0
)

row = final_df.iloc[index]

st.metric("ML Risk Score", round(row["ml_risk"], 3))
st.metric("Text Risk Score", round(row["text_risk"], 3))
st.metric("Sequence Risk Score", round(row["seq_risk"], 3))

st.markdown("### Final Threat Level")

if row["final_threat_score"] > 70:
    st.error(f"HIGH THREAT ({row['final_threat_score']:.2f})")
elif row["final_threat_score"] > 40:
    st.warning(f"MEDIUM THREAT ({row['final_threat_score']:.2f})")
else:
    st.success(f"LOW THREAT ({row['final_threat_score']:.2f})")
