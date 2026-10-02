# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURES = ['duration', 'packets', 'bytes', 'avg_pkt_size', 'src_port_entropy', 'dst_port_entropy',
            'protocol_0', 'protocol_1', 'protocol_2']

def prepare_df(df: pd.DataFrame):
    # One hot for protocol
    proto = pd.get_dummies(df['protocol'], prefix='protocol')
    df = pd.concat([df.drop(columns=['protocol']), proto], axis=1)
    # ensure columns exist
    for c in ['protocol_0','protocol_1','protocol_2']:
        if c not in df.columns:
            df[c]=0
    return df[sorted(df.columns)]

def fit_scaler(df: pd.DataFrame):
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler

def scale_df(df: pd.DataFrame, scaler):
    return pd.DataFrame(scaler.transform(df), columns=df.columns)

