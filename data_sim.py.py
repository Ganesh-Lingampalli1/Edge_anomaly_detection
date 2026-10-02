import pandas as pd
import numpy as np

def generate_normal(n=100):
    return pd.DataFrame({
        'latency': np.random.normal(50, 10, n),
        'throughput': np.random.normal(100, 20, n),
        'packet_loss': np.random.uniform(0.0, 0.2, n),
        'cpu_usage': np.random.uniform(20, 60, n)
    })

def generate_anomalies(n=20):
    return pd.DataFrame({
        'latency': np.random.normal(150, 40, n),
        'throughput': np.random.normal(40, 15, n),
        'packet_loss': np.random.uniform(0.3, 0.9, n),
        'cpu_usage': np.random.uniform(80, 100, n)
    })
