import os, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics.gravimetry import GravityModelling2D

# ============================================================
# PARAMETERS
# ============================================================
class Args: pass
args = Args()

args.out_dir      = "MTP_Gravity_ANN"
args.seed         = 42

args.xmax         = 76000.0
args.ymax         = 10000.0
args.cell_area    = 5e5

args.rho1_min     = -500.0
args.rho1_max     = -400.0
args.rho2_min     = -200.0
args.rho2_max     =  50.0
args.rho3_fixed   =  110.0

args.n_samples      = 3000
args.noise_std_mgal = 0.05

args.epochs        = 50
args.batch_size    = 64
args.lr            = 1e-3
args.weight_decay  = 1e-4
args.hidden_width  = 1024
args.hidden_layers = 4
args.dropout       = 0.05

args.lambda_phys   = 2.0
args.lambda_smooth = 1e-4
args.lambda_mag    = 1e-7

args.refine_iters  = 1500
args.lambda_refine = 1e-2

args.obs_csv       = "gravity_data.csv"
args.obs_x_unit    = "km"
args.g_unit_label  = "mGal"

# ============================================================
# SETUP
# ============================================================
os.makedirs(args.out_dir, exist_ok=True)
fig_dir = os.path.join(args.out_dir, "figures")
res_dir = os.path.join(args.out_dir, "results")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MESH
# ============================================================
rect = mt.createPolygon(
    [[0,0],[args.xmax,0],[args.xmax,-args.ymax],[0,-args.ymax]],
    isClosed=True
)
mesh = mt.createMesh(rect, quality=34, area=args.cell_area)
M = mesh.cellCount()

centers = np.array([[c.center().x(), -c.center().y()] for c in mesh.cells()])
cx = centers[:,0]
cz_neg = -centers[:,1]

# ============================================================
# SEISMIC HORIZONS (EXACT)
# ============================================================
def _interp_fun(pts):
    pts = np.array(pts, float)
    xs, zs = pts[:,0], pts[:,1]
    srt = np.argsort(xs)
    return lambda xq: np.interp(xq, xs[srt], zs[srt])

scale_x = args.xmax / 76.0
scale_z = args.ymax / 10.0

h1_pts = [
    [0, -0.25*scale_z],[6*scale_x, -0.25*scale_z],[10*scale_x, -1.05*scale_z],
    [50*scale_x, -1.05*scale_z],[53*scale_x, -0.75*scale_z],[60*scale_x, -0.73*scale_z],
    [63*scale_x, -0.62*scale_z],[65*scale_x, 0],[70*scale_x, 0]
]

h2_pts = [
    [0, -1.8*scale_z],[5*scale_x, -1.8*scale_z],[8.5*scale_x, -2.3*scale_z],
    [10.8*scale_x, -2.1*scale_z],[20*scale_x, -5.75*scale_z],[27.5*scale_x, -5.0*scale_z],
    [34*scale_x, -6.0*scale_z],[40*scale_x, -6.8*scale_z],[48*scale_x, -5.6*scale_z],
    [52*scale_x, -2.5*scale_z],[64*scale_x, -2.5*scale_z],[70*scale_x, -2.7*scale_z]
]

z1 = _interp_fun(h1_pts)(cx)
z2 = _interp_fun(h2_pts)(cx)

mask1 = (cz_neg <= 0) & (cz_neg >= z1)
mask2 = (cz_neg < z1) & (cz_neg >= z2)
mask3 = cz_neg < z2

idx1 = np.where(mask1)[0]
idx2 = np.where(mask2)[0]
idx3 = np.where(mask3)[0]
idx_free = np.concatenate([idx1, idx2])

M1, M2, M3 = len(idx1), len(idx2), len(idx3)
print(f"Cells: L1={M1}, L2={M2}, L3={M3}, total={M}")

# ============================================================
# OBSERVED DATA
# ============================================================
def read_x_and_g(csv):
    df = pd.read_csv(csv, sep=None, engine="python")
    x = df.iloc[:,0].values
    g = df.iloc[:,1].values
    if args.obs_x_unit == "km":
        x *= 1000
    return x, g

x_stations, g_obs = read_x_and_g(args.obs_csv)

# ============================================================
# FORWARD OPERATOR
# ============================================================
def build_A(x):
    pts = np.c_[x, np.zeros_like(x)]
    fop = GravityModelling2D(mesh=mesh, points=pts)
    A = np.zeros((len(x), M))
    unit = np.zeros(M)
    for j in range(M):
        unit[:] = 0
        unit[j] = 1
        A[:,j] = fop.response(unit)
    return A

A_np = build_A(x_stations)
A_torch = torch.tensor(A_np, dtype=torch.float32, device=device)

# ============================================================
# ASSEMBLY
# ============================================================
def assemble_rho(r_free):
    rho = torch.zeros(r_free.shape[0], M, device=device)
    rho[:, idx1] = r_free[:, :M1]
    rho[:, idx2] = r_free[:, M1:]
    rho[:, idx3] = args.rho3_fixed
    return rho

# ============================================================
# SMOOTHNESS
# ============================================================
x_order = np.argsort(cx)
nbr_i = torch.tensor(x_order[:-1], device=device)
nbr_j = torch.tensor(x_order[1:], device=device)

def smoothness_loss(rho):
    return ((rho[:,nbr_i] - rho[:,nbr_j])**2).mean()

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================
def smooth_x(v, x, L=6000):
    w = np.exp(-0.5*((x[:,None]-x[None,:])/L)**2)
    w /= w.sum(axis=1, keepdims=True)
    return w @ v

def random_model():
    rho = np.full(M, args.rho3_fixed)
    rho[idx1] = smooth_x(
        np.random.uniform(args.rho1_min, args.rho1_max, M1),
        cx[idx1]
    )
    rho[idx2] = smooth_x(
        np.random.uniform(args.rho2_min, args.rho2_max, M2),
        cx[idx2]
    )
    return rho

def sample_batch(n):
    R, G = [], []
    for _ in range(n):
        rho = random_model()
        g = A_np @ rho + np.random.normal(0, args.noise_std_mgal, len(x_stations))
        R.append(rho[idx_free])
        G.append(g)
    return (
        torch.tensor(G, dtype=torch.float32, device=device),
        torch.tensor(R, dtype=torch.float32, device=device)
    )

# ============================================================
# ANN
# ============================================================
class InvNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        last = len(x_stations)
        for _ in range(args.hidden_layers):
            layers += [
                nn.Linear(last, args.hidden_width),
                nn.ReLU(),
                nn.Dropout(args.dropout)
            ]
            last = args.hidden_width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(last, len(idx_free))

    def forward(self, g):
        raw = self.head(self.body(g))
        r1 = args.rho1_min + (args.rho1_max - args.rho1_min) * torch.sigmoid(raw[:,:M1])
        r2 = args.rho2_min + (args.rho2_max - args.rho2_min) * torch.sigmoid(raw[:,M1:])
        return torch.cat([r1, r2], dim=1)

net = InvNet().to(device)
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# ============================================================
# NORMALIZATION
# ============================================================
with torch.no_grad():
    Gs, _ = sample_batch(512)
g_mu = Gs.mean(0)
g_std = Gs.std(0).clamp_min(1e-6)
def norm_g(g): return (g - g_mu) / g_std

# ============================================================
# TRAINING
# ============================================================
train_loss_hist = []
rmse_model_hist = []
rmse_data_hist  = []

steps = max(1, args.n_samples // args.batch_size // 4)
print("\n=== TRAINING ===")

for ep in range(1, args.epochs + 1):
    net.train()
    total = 0.0

    for _ in range(steps):
        g, r = sample_batch(args.batch_size)
        rp = net(norm_g(g))
        rho = assemble_rho(rp)
        gp = torch.matmul(rho, A_torch.T)

        loss = (
            F.mse_loss(rp, r)
            + args.lambda_phys * F.mse_loss(gp, g)
            + args.lambda_smooth * smoothness_loss(rho)
            + args.lambda_mag * (rp**2).mean()
        )

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()

    net.eval()
    with torch.no_grad():
        g, r = sample_batch(8)
        rp = net(norm_g(g))
        rho = assemble_rho(rp)
        gp = torch.matmul(rho, A_torch.T)
        rm = F.mse_loss(rp, r).sqrt().item()
        rd = F.mse_loss(gp, g).sqrt().item()

    train_loss_hist.append(total / steps)
    rmse_model_hist.append(rm)
    rmse_data_hist.append(rd)



# ============================================================
# ANN INVERSION
# ============================================================
net.eval()
with torch.no_grad():
    r_free = net(norm_g(torch.tensor(g_obs, dtype=torch.float32, device=device)[None,:]))
    rho_ann = assemble_rho(r_free)[0].cpu().numpy()

np.save(os.path.join(res_dir, "rho_ann.npy"), rho_ann)

def refine(rho, g, A, lam, niter):
    Af = A[:, idx_free]
    I = np.eye(len(idx_free))
    rms = []

    for k in range(niter):
        r = g - A @ rho
        rms.append(np.sqrt(np.mean(r**2)))
        d = np.linalg.solve(Af.T @ Af + lam * I, Af.T @ r)
        rho[idx_free] += d
        rho[idx1] = np.clip(rho[idx1], args.rho1_min, args.rho1_max)
        rho[idx2] = np.clip(rho[idx2], args.rho2_min, args.rho2_max)
        rho[idx3] = args.rho3_fixed

    return rho, np.array(rms)

rho_final, rms_iter = refine(
    rho_ann.copy(),
    g_obs,
    A_np,
    args.lambda_refine,
    args.refine_iters
)

np.save(os.path.join(res_dir, "rho_final.npy"), rho_final)
np.save(os.path.join(res_dir, "iterative_rms.npy"), rms_iter)


# ============================================================
# FINAL GRAVITY FIT
# ============================================================
g_final = A_np @ rho_final

np.savez(
    os.path.join(res_dir, "gravity_fit.npz"),
    x_km=x_stations/1000,
    g_obs=g_obs,
    g_final=g_final
)

plt.figure(figsize=(10,4))
plt.plot(x_stations/1000, g_obs, "k.", ms=3, label="Observed")
plt.plot(x_stations/1000, g_final, "b-", lw=2, label="ANN")
plt.xlabel("x [km]")
plt.ylabel(f"Δg [{args.g_unit_label}]")
plt.title("Gravity Anomaly Fit")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "gravity_data_fit.png"), dpi=300)
plt.show()

# ============================================================
# FINAL DENSITY MODEL
# ============================================================
vmax = max(150, np.percentile(np.abs(rho_final), 98))

fig, ax = plt.subplots(figsize=(7,4))
pg.show(
    mesh,
    rho_final,
    cMap="seismic",
    vmin=-vmax,
    vmax=vmax,
    label="Density contrast (kg/m³)",
    ax=ax
)
ax.set_title("Inverted Density Model")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "inverted_density_model.png"), dpi=800)
plt.show()



# ============================================================
# FINAL GRAVITY FIT
# ============================================================
g_final = A_np @ rho_final

np.savez(
    os.path.join(res_dir, "gravity_fit.npz"),
    x_km=x_stations/1000,
    g_obs=g_obs,
    g_final=g_final
)

plt.figure(figsize=(10,4))
plt.plot(x_stations/1000, g_obs, "k.", ms=3, label="Observed")
plt.plot(x_stations/1000, g_final, "b-", lw=2, label="ANN")
plt.xlabel("x [km]")
plt.ylabel(f"Δg [{args.g_unit_label}]")
plt.title("Gravity Anomaly Fit")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "gravity_data_fit.png"), dpi=300)
plt.show()

# ============================================================
# FINAL DENSITY MODEL
# ============================================================
vmax = max(150, np.percentile(np.abs(rho_final), 98))

fig, ax = plt.subplots(figsize=(7,4))
pg.show(
    mesh,
    rho_final,
    cMap="seismic",
    vmin=-vmax,
    vmax=vmax,
    label="Density contrast (kg/m³)",
    ax=ax
)
ax.set_title("Inverted Density Model")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "inverted_density_model.png"), dpi=800)
plt.show()




# ============================================================
# FINAL GRAVITY FIT + DENSITY MODEL (COMBINED FIGURE)
# ============================================================

g_final = A_np @ rho_final

np.savez(
    os.path.join(res_dir, "gravity_fit.npz"),
    x_km=x_stations/1000,
    g_obs=g_obs,
    g_final=g_final
)

vmax = max(150, np.percentile(np.abs(rho_final), 98))

# ---- Create combined figure ----
fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(10, 8),
    gridspec_kw={'height_ratios': [1, 2.5]}
)

# ------------------------------------------------------------
# TOP: Gravity anomaly fit
# ------------------------------------------------------------
ax1.plot(x_stations, g_obs, "k*", ms=7, label="Observed")
ax1.plot(x_stations, g_final, "b", lw=2, label="ANN")
ax1.set_xlabel("x [m]")
ax1.set_ylabel(f"Δg [{args.g_unit_label}]")
ax1.set_title("Gravity Anomaly Fit")
ax1.legend()
ax1.set_xlim(0, 76000)
ax1.grid(alpha=0.3)

# ------------------------------------------------------------
# BOTTOM: Inverted density model
# ------------------------------------------------------------
pg.show(
    mesh,
    rho_final,
    cMap="viridis",
    vmin=-vmax,
    vmax=vmax,
    showMesh=False,   # <-- comma was missing here
    label="Density contrast (kg/m³)",
    ax=ax2
)

ax2.set_title("Inverted Density Model")

plt.tight_layout()
plt.savefig(
    os.path.join(fig_dir, "gravity_fit_and_density_model.png"),
    dpi=800
)
plt.show()