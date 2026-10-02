# train_autoencoder.py
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from data_sim import generate_normal

print("Generating training data...")

# ----------------------------
# Generate normal data
# ----------------------------
df = generate_normal(2000)

features = ['latency', 'throughput', 'packet_loss', 'cpu_usage']
X = df[features].values

# ----------------------------
# Scale data
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.joblib")
print("Saved scaler.joblib")

# ----------------------------
# Build autoencoder
# ----------------------------
input_dim = X_scaled.shape[1]

inp = Input(shape=(input_dim,))
x = Dense(8, activation="relu")(inp)
x = Dense(4, activation="relu")(x)
x = Dense(8, activation="relu")(x)
out = Dense(input_dim)(x)

model = Model(inputs=inp, outputs=out)
model.compile(optimizer="adam", loss="mse")

print("Training autoencoder...")
model.fit(X_scaled, X_scaled, epochs=10, batch_size=32, verbose=1)

# ----------------------------
# Save model
# ----------------------------
model.save("autoencoder_model.keras")
print("Saved autoencoder_model.keras")

# ----------------------------
# Compute threshold
# ----------------------------
recon = model.predict(X_scaled)
mse = np.mean((recon - X_scaled) ** 2, axis=1)
threshold = float(np.percentile(mse, 99))

joblib.dump(threshold, "threshold.joblib")
print("Saved threshold.joblib:", threshold)

print("Training complete ✅")