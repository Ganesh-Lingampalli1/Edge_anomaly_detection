# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from preprocess import prepare_df, scale_df
import tensorflow as tf
import plotly.express as px

st.set_page_config(layout="wide", page_title="Edge Anomaly Detection")

st.title("Real-time Edge Anomaly Detection")
st.markdown("Streaming simulated network flows, showing anomaly scores from an autoencoder.")

# ----------------------------
# Load model, scaler, threshold
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("autoencoder_model.keras")
    scaler = joblib.load("scaler.joblib")
    threshold = float(joblib.load("threshold.joblib"))
    return model, scaler, threshold

model, scaler, threshold = load_artifacts()

# ----------------------------
# UI Controls
# ----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Stream Controls")
    speed = st.slider("Flows per second (simulation)", min_value=1, max_value=200, value=30)
    anomaly_rate = st.slider("Injected anomaly rate (%)", 0, 100, 5)
    run = st.button("Start Stream")
    st.write("Threshold (recon MSE):", round(threshold, 6))

with col2:
    st.header("Recent Flows (last 200)")
    
    flow_view = st.empty()

# ----------------------------
# Storage
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=['latency', 'throughput', 'packet_loss', 'cpu_usage', 'mse', 'is_anomaly']
    )

# ----------------------------
# Import data simulator
# ----------------------------
from data_sim import generate_normal, generate_anomalies

def simulate_batch(n, anomaly_rate_percent):
    n_anom = int(n * anomaly_rate_percent / 100)
    n_norm = n - n_anom
    df_norm = generate_normal(n_norm)
    df_anom = generate_anomalies(n_anom)
    df = pd.concat([df_norm, df_anom], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

# ----------------------------
# Stream control buttons
# ----------------------------
if run:
    st.session_state.running = True

if st.session_state.get("running", False):
    stop = st.button("Stop Stream")
    if stop:
        st.session_state.running = False

# ----------------------------
# Streaming loop
# ----------------------------
if st.session_state.get("running", False):
    placeholder = st.empty()
    progress = st.progress(0)
    i = 0
    try:
        while st.session_state.running:
            i += 1

            # Simulate incoming network flow batch
            batch = simulate_batch(max(1, speed // 2), anomaly_rate)

            # ----------------------------
            # Use only model features
            # ----------------------------
            model_cols = ['latency', 'throughput', 'packet_loss', 'cpu_usage']

            missing = [c for c in model_cols if c not in batch.columns]
            if missing:
                st.error(f"Streaming error: missing columns for model input: {missing}")
                st.session_state.running = False
                break

            df_model = batch[model_cols].copy()

            # Scale & predict
            df_scaled = scale_df(df_model, scaler)
            preds = model.predict(df_scaled)
            mse = np.mean(np.square(preds - df_scaled.values), axis=1)
            is_anom = mse > threshold

            # Attach results
            batch['mse'] = mse
            batch['is_anomaly'] = is_anom

            # Update history
            st.session_state.history = pd.concat(
                [batch, st.session_state.history], ignore_index=True
            ).head(200)

            # ----------------------------
            # Display visualization
            # ----------------------------
            hist = st.session_state.history.copy()
            hist['time'] = range(len(hist))

            fig = px.scatter(
                hist,
                x='time',
                y='mse',
                color=hist['is_anomaly'].map({True: 'anomaly', False: 'normal'}),
                hover_data=['latency', 'throughput', 'packet_loss', 'cpu_usage']
            )

            with placeholder.container():
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    hist[['latency', 'throughput', 'packet_loss', 'cpu_usage', 'mse', 'is_anomaly']].head(50)
                )

            progress.progress(min(100, int((i % 100) / 100 * 100)))
            time.sleep(max(0.01, 1.0 / speed))

    except Exception as e:
        st.error(f"⚠️ Error during streaming: {e}")

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    st.write("✅ Streamlit app loaded successfully. Use the Start Stream button to begin.")
