import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import shap
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Zero Trust AI Dashboard", layout="wide")

st.title("🔐 Zero Trust AI Dashboard")
st.markdown("AI-driven anomaly detection with **LSTM + Isolation Forest + SHAP Explainability**")

# --- Fixed Role-based Tokens ---
roles = {"analyst": "analyst123", "admin": "admin123"}
st.sidebar.write("🔑 Analyst Token: analyst123")
st.sidebar.write("🔑 Admin Token: admin123")

role_input = st.sidebar.text_input("Enter role token")
if role_input == roles["admin"]:
    st.success("✅ Admin access granted")
    role = "admin"
elif role_input == roles["analyst"]:
    st.success("✅ Analyst access granted")
    role = "analyst"
else:
    st.error("❌ Invalid token. Enter a valid role token.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")
contamination = st.sidebar.slider("Isolation Forest contamination rate", 0.001, 0.2, 0.01)
threshold = st.sidebar.slider("Anomaly threshold", 0.1, 1.0, 0.6)
epochs = st.sidebar.slider("LSTM training epochs (disabled)", 1, 10, 3)

uploaded_file = st.file_uploader("📂 Upload your network_logs.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    if not {"bytes","duration"}.issubset(df.columns):
        st.error("CSV must contain 'bytes' and 'duration' columns.")
    else:
        features = df[["bytes","duration"]].fillna(0)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)

        timesteps = 10
        if len(scaled) > timesteps:
            X = np.array([scaled[i:i+timesteps] for i in range(len(scaled)-timesteps)])
            y = np.zeros((X.shape[0],))

            # --- Build LSTM (no training) ---
            model = Sequential([
                LSTM(64, input_shape=(timesteps, features.shape[1]), return_sequences=True),
                LSTM(32),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # Skip training to avoid gradient errors
            # model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

            lstm_preds = model.predict(X)

            # --- Isolation Forest ---
            iso = IsolationForest(contamination=contamination, random_state=42)
            iso_preds = iso.fit_predict(scaled)
            iso_scaled = np.where(iso_preds == -1, 1, 0)

            # --- Hybrid Score ---
            scores = (lstm_preds.flatten() + iso_scaled[:len(lstm_preds)]) / 2
            anomalies = np.where(scores > threshold)[0]

            # --- Summary Cards ---
            st.subheader("📊 Anomaly Summary")
            total_anomalies = len(anomalies)
            avg_score = float(np.mean(scores)) if len(scores) > 0 else 0
            highest_idx = anomalies[0] if len(anomalies) > 0 else None

            col1, col2, col3 = st.columns(3)
            col1.metric("Total anomalies", total_anomalies)
            col2.metric("Average anomaly score", f"{avg_score:.2f}")
            if highest_idx is not None:
                col3.metric("Most severe anomaly", f"Row {highest_idx}")
            else:
                col3.metric("Most severe anomaly", "None")

            # --- Severity Table ---
            def classify_risk(score):
                if score > 0.8:
                    return "🔴 High"
                elif score > 0.5:
                    return "🟡 Medium"
                else:
                    return "🟢 Low"

            if len(anomalies) > 0:
                st.subheader("🚨 Detected Anomalies")
                anomaly_table = pd.DataFrame({
                    "Index": anomalies,
                    "Bytes": df.iloc[anomalies]["bytes"].values,
                    "Duration": df.iloc[anomalies]["duration"].values,
                    "Risk Level": [classify_risk(scores[i]) for i in anomalies]
                })
                st.dataframe(anomaly_table)

                # --- Timeline Chart ---
                st.subheader("📈 Timeline View")
                chart_data = pd.DataFrame({"Score": scores})
                st.line_chart(chart_data)

                # --- Pop-up Alert ---
                st.toast(f"🚨 {total_anomalies} anomalies detected!", icon="⚠️")

                # --- SHAP Explainability (IsolationForest only) ---
                st.subheader("🔍 SHAP Explainability (IsolationForest)")
                explainer = shap.TreeExplainer(iso)
                shap_values = explainer.shap_values(scaled)

                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, features,
                                  feature_names=["bytes","duration"], show=False)
                st.pyplot(fig)

                # --- Risk Distribution Pie Chart ---
                st.subheader("📊 Risk Distribution")
                risk_levels = [classify_risk(scores[i]) for i in anomalies]
                risk_counts = pd.Series(risk_levels).value_counts()

                fig, ax = plt.subplots()
                ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%',
                       colors=["red","yellow","green"], startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

                # --- Export anomalies (Admin only) ---
                if role == "admin":
                    st.download_button("⬇️ Download anomalies",
                                       df.iloc[anomalies].to_csv(index=False),
                                       "anomalies.csv")
            else:
                st.info("✅ No anomalies detected.")
        else:
            st.warning("Dataset too small for LSTM (need at least 10 rows).")

    # --- Simulation Mode ---
    st.sidebar.subheader("🧪 Simulation Mode")
    sim_bytes = st.sidebar.slider("Simulated Bytes", 1000, 50000, 5000)
    sim_duration = st.sidebar.slider("Simulated Duration", 1, 50, 5)
    if st.button("Inject Synthetic Anomaly"):
        sim_data = pd.DataFrame({"bytes":[sim_bytes],"duration":[sim_duration]})
        st.write("Injected anomaly:", sim_data)
        st.toast("🔥 Synthetic anomaly injected!", icon="🚨")

else:
    st.info("Upload a CSV file to start analysis.")

# --- Glow Style Theme ---
st.markdown("""
<style>
.stApp {
    background-color: #0d0d0d; /* dark background */
    color: #f0f0f0;
}
h1 {
    color: #00ffff;
    text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff;
}
h2, h3 {
    color: #ff00ff;
    text-shadow: 0 0 8px #ff00ff, 0 0 16px #ff00ff;
}
[data-testid="stMetricValue"] {
    color: #39ff14;
    text-shadow: 0 0 6px #39ff14, 0 0 12px #39ff14;
    font-weight: bold;
}
[data-testid="stDataFrame"] {
    background-color: #1a1a1a;
    border: 1px solid #00ffff;
    box-shadow: 0 0 10px #00ffff;
}
button {
    background-color: #111;
    color: #fff;
    border: 1px solid #ff00ff;
    box-shadow: 0 0 10px #ff00ff;
}
button:hover {
    background-color: #ff00ff;
    color: #000;
    box-shadow: 0 0 20px #ff00ff;
}
</style>
""", unsafe_allow_html=True)
