import os, random, time
import numpy as np
import torch
import pandas as pd


def set_seed(seed):
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)




def read_x_and_g_strict(csv_path, obs_x_unit="km"):
df = pd.read_csv(csv_path, comment="#", sep=None, engine="python")
num_cols = [c for c in df.columns if np.issubdtype(df[c].dropna().dtype, np.number)]
x = df[num_cols[0]].to_numpy(float)
g = df[num_cols[1]].to_numpy(float)
mask = np.isfinite(x) & np.isfinite(g)
x, g = x[mask], g[mask]
if obs_x_unit.lower() == "km":
x *= 1000.0
return x, g




def format_seconds(s):
if s < 60: return f"{s:.1f}s"
m, s = divmod(s, 60)
if m < 60: return f"{int(m)}m {int(s)}s"
h, m = divmod(int(m), 60)
return f"{h}h {m}m {int(s)}s"