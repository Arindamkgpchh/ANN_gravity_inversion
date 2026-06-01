import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics.gravimetry import GravityModelling2D
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ── publication style ────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif"],
    "font.size":            11,
    "axes.linewidth":       0.8,
    "axes.titlesize":       12,
    "axes.titleweight":     "bold",
    "axes.labelsize":       11,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "xtick.major.size":     5,
    "xtick.minor.size":     2.5,
    "ytick.major.size":     5,
    "ytick.minor.size":     2.5,
    "legend.framealpha":    0.9,
    "legend.edgecolor":     "0.7",
    "legend.fontsize":      9,
    "figure.dpi":           150,
    "savefig.dpi":          600,
    "savefig.bbox":         "tight",
})

CMAP    = "viridis"
out_dir = "AGM_ANN_pub"
os.makedirs(out_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GEOMETRY & SHARED MESH
# ─────────────────────────────────────────────────────────────────────────────
xmax, ymax = 75_000.0, 10_000.0
xf         = 55_000.0

h1 = [[0,     -2000], [10000, -2400], [25000, -2100],
      [40000, -2001], [75000, -2570]]

h2 = [[0,     -4600], [12500, -5000], [25000, -4800],
      [37500, -4400], [75000, -5000]]

world              = mt.createPolygon([[0,0],[xmax,0],[xmax,-ymax],[0,-ymax]],
                                       isClosed=True, marker=0)
top_polygon        = mt.createPolygon([[0,0]] + h1 + [[xmax,0]],
                                       isClosed=True, marker=1)
middle_polygon     = mt.createPolygon(h1 + h2[::-1],
                                       isClosed=True, marker=2)
h2_x, h2_y        = zip(*h2)
y_xf               = float(np.interp(xf, h2_x, h2_y))

pts_bl = [p for p in h2 if p[0] <= xf] + [[xf, y_xf],[xf,-ymax],[0,-ymax]]
bottom_left_polygon  = mt.createPolygon(pts_bl, isClosed=True, marker=3)

pts_br = [[xf, y_xf]] + [p for p in h2 if p[0] >= xf] + [[xmax,-ymax],[xf,-ymax]]
bottom_right_polygon = mt.createPolygon(pts_br, isClosed=True, marker=4)

geom        = world + top_polygon + middle_polygon + bottom_left_polygon + bottom_right_polygon
mesh_shared = mt.createMesh(geom, quality=34, area=1_000_000.0)

# measurement profile
x    = np.arange(0, 75_000.0, 1_000.0)
pnts = np.vstack((x, np.zeros_like(x))).T
fop  = GravityModelling2D(mesh=mesh_shared, points=pnts)

h1_x, h1_y = zip(*h1)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PRE-COMPUTE MESH NODE COORDS & TRIANGLE CONNECTIVITY
#     Uses pg.x() / pg.y() + cell.node(j).id()  — confirmed working approach
# ─────────────────────────────────────────────────────────────────────────────
xn = np.array(pg.x(mesh_shared))   # shape (nNodes,)
yn = np.array(pg.y(mesh_shared))   # shape (nNodes,)

triangles = np.array(
    [[mesh_shared.cell(i).node(j).id() for j in range(mesh_shared.cell(i).nodeCount())]
     for i in range(mesh_shared.cellCount())]
)   # shape (nCells, 3)


def draw_mesh(ax, cell_values, cmap=CMAP, vmin=None, vmax=None):
    """Render unstructured pyGIMLi mesh using matplotlib tripcolor (correct & fast)."""
    tc = ax.tripcolor(xn / 1e3, yn / 1e3, triangles,
                      facecolors=cell_values,
                      cmap=cmap, vmin=vmin, vmax=vmax,
                      shading="flat", rasterized=True)
    return tc

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FORWARD HELPER
# ─────────────────────────────────────────────────────────────────────────────
def forward_from_params(densities):
    mc = np.zeros(mesh_shared.cellCount())
    for cell in mesh_shared.cells():
        mk = cell.marker()
        if 1 <= mk <= 4:
            mc[cell.id()] = densities[mk - 1]
        else:
            mc[cell.id()] = float(np.mean(densities))
    return np.asarray(fop.response(mc)), mc

# ─────────────────────────────────────────────────────────────────────────────
# 4.  ANN DATASET
# ─────────────────────────────────────────────────────────────────────────────
ranges    = [(-225,-175), (-120,-80), (20,70), (275,325)]
n_samples = 15_000
rng       = np.random.default_rng(42)

print("Generating training data …")
X_data = np.zeros((n_samples, len(pnts)))
Y_data = np.zeros((n_samples, 4))

for i in range(n_samples):
    params    = [float(rng.uniform(lo, hi)) for lo, hi in ranges]
    g, _      = forward_from_params(params)
    noise     = 0.10 * np.std(g) * rng.standard_normal(g.shape)
    X_data[i] = g + noise
    Y_data[i] = params

X_train, X_test, Y_train, Y_test = train_test_split(
    X_data, Y_data, test_size=0.2, random_state=42)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  ANN TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("Training ANN …")
nn = MLPRegressor(hidden_layer_sizes=(256,128,64),
                  activation="relu", max_iter=5000,
                  tol=1e-4, random_state=42, verbose=True)
nn.fit(X_train, Y_train)

train_rmse = np.sqrt(mean_squared_error(Y_train, nn.predict(X_train)))
test_rmse  = np.sqrt(mean_squared_error(Y_test,  nn.predict(X_test)))
print(f"\nTrain RMSE: {train_rmse:.4f}   Test RMSE: {test_rmse:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRUE MODEL & OBSERVED DATA
# ─────────────────────────────────────────────────────────────────────────────
true_params      = [-200, -100, 50, 300]
g_clean, true_mc = forward_from_params(true_params)
g_obs            = g_clean + 0.05 * np.std(g_clean) * rng.standard_normal(g_clean.shape)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  ANN INVERSION  (same mesh, marker-based assignment)
# ─────────────────────────────────────────────────────────────────────────────
pred_params = nn.predict([g_obs])[0]
for i, (lo, hi) in enumerate(ranges):
    pred_params[i] = float(np.clip(pred_params[i], lo, hi))

print(f"\nTrue params  : {true_params}")
print(f"Predicted    : {np.round(pred_params,2).tolist()}")

inv_mc = np.zeros(mesh_shared.cellCount())
for cell in mesh_shared.cells():
    mk = cell.marker()
    if 1 <= mk <= 4:
        inv_mc[cell.id()] = pred_params[mk - 1]
    else:
        inv_mc[cell.id()] = float(np.mean(pred_params))

g_pred = np.asarray(fop.response(inv_mc))
misfit = g_obs - g_pred
rms    = np.sqrt(np.mean(misfit**2))
print(f"Inversion RMS misfit: {rms:.6f} mGal")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  NOISE SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────
noise_levels = [0, 5, 10, 15, 20, 40, 50]
rms_values, chi2_values, model_errors = [], [], []

print("\n─── Noise sensitivity ───")
for nl in noise_levels:
    g_c, _ = forward_from_params(true_params)
    sigma   = (nl/100.0)*np.std(g_c) if nl > 0 else 0.01*np.std(g_c)
    g_n     = g_c + sigma * rng.standard_normal(g_c.shape)

    pp = nn.predict([g_n])[0]
    for i,(lo,hi) in enumerate(ranges):
        pp[i] = float(np.clip(pp[i], lo, hi))

    mc_n = np.zeros(mesh_shared.cellCount())
    for cell in mesh_shared.cells():
        mk = cell.marker()
        if 1 <= mk <= 4:
            mc_n[cell.id()] = pp[mk-1]
        else:
            mc_n[cell.id()] = float(np.mean(pp))

    gp   = np.asarray(fop.response(mc_n))
    res  = g_n - gp
    rms_ = np.sqrt(np.mean(res**2))
    chi2 = np.mean((res/sigma)**2)
    merr = np.sqrt(np.mean((np.array(true_params) - pp)**2))

    rms_values.append(rms_);  chi2_values.append(chi2);  model_errors.append(merr)
    print(f"  {nl:3d}% → RMS={rms_:.4f}  χ²={chi2:.3f}  ModelErr={merr:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# ── PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
x_km = x / 1e3
vmin = min(true_params) * 1.05
vmax = max(true_params) * 1.05


def draw_horizons(ax, lw=1.6, alpha=0.88):
    h1a = np.array(h1);  h2a = np.array(h2)
    l1, = ax.plot(h1a[:,0]/1e3, h1a[:,1]/1e3, "-",  color="white", lw=lw, alpha=alpha)
    l2, = ax.plot(h2a[:,0]/1e3, h2a[:,1]/1e3, "--", color="white", lw=lw, alpha=alpha)
    ax.axvline(xf/1e3, color="white", lw=1.0, ls=":", alpha=0.65)
    return l1, l2


def style_model_ax(ax):
    ax.set_xlim(0, xmax/1e3)
    ax.set_ylim(-ymax/1e3, 0)
    ax.set_ylabel("Depth (km)", labelpad=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))


def style_anom_ax(ax):
    ax.set_xlim(0, xmax/1e3)
    ax.set_ylabel("Δg (mGal)", labelpad=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(labelbottom=False)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 – TRAINING CONVERGENCE
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.2))
iters = np.arange(1, len(nn.loss_curve_)+1)
ax.semilogy(iters, nn.loss_curve_, color="#c0392b", lw=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss (log scale)")
ax.set_title("ANN Training Convergence Curve")
ax.annotate(f"Final loss = {nn.loss_curve_[-1]:.4e}",
            xy=(iters[-1], nn.loss_curve_[-1]),
            xytext=(0.55, 0.70), textcoords="axes fraction",
            fontsize=9, color="0.25",
            arrowprops=dict(arrowstyle="->", lw=0.8, color="0.4"))
ax.grid(which="major", ls="--", lw=0.5, color="0.80")
ax.grid(which="minor", ls=":",  lw=0.3, color="0.90")
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.text(0.02, 0.06,
        f"Train RMSE = {train_rmse:.4f}\nTest RMSE  = {test_rmse:.4f}",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="0.97", ec="0.7", lw=0.7))
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig1_training_convergence.png"))
plt.close()
print("Saved fig1_training_convergence.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – SYNTHETIC MODEL
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 7.5))
gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 2.5], hspace=0.05,
                        left=0.10, right=0.87, top=0.94, bottom=0.08)

ax_ga = fig.add_subplot(gs[0])
ax_ga.plot(x_km, g_obs,   color="#1a1a2e", lw=1.5, label="Observed (5% noise)")
ax_ga.plot(x_km, g_clean, color="#e94560", lw=1.3, ls="--", label="Noise-free")
ax_ga.fill_between(x_km, g_obs, g_clean, alpha=0.15, color="#e94560")
style_anom_ax(ax_ga)
ax_ga.set_title("(a) Synthetic Forward Model", pad=7)
ax_ga.legend(loc="upper left", fontsize=9)

ax_gm = fig.add_subplot(gs[1])
tc = draw_mesh(ax_gm, true_mc, vmin=vmin, vmax=vmax)
l1, l2 = draw_horizons(ax_gm)
style_model_ax(ax_gm)
ax_gm.set_xlabel("Distance (km)", labelpad=4)
ax_gm.legend([l1, l2], ["Horizon 1", "Horizon 2"],
             loc="lower right", title="Interfaces", title_fontsize=8, fontsize=8)
for lbl, xp, yp in [("Layer 1",37.5,-1.0),("Layer 2",37.5,-3.5),
                     ("Layer 3",27.5,-7.5),("Layer 4",65.0,-7.5)]:
    ax_gm.text(xp, yp, lbl, ha="center", va="center",
               color="white", fontsize=9, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.40, ec="none"))

cax = fig.add_axes([0.88, 0.08, 0.022, 0.86])
cb  = fig.colorbar(tc, cax=cax)
cb.set_label("Density Contrast (kg/m³)", labelpad=8)
cb.ax.yaxis.set_minor_locator(AutoMinorLocator(4))

plt.savefig(os.path.join(out_dir, "fig2_synthetic_model.png"))
plt.close()
print("Saved fig2_synthetic_model.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 – INVERTED MODEL
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 7.5))
gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 2.5], hspace=0.05,
                        left=0.10, right=0.87, top=0.94, bottom=0.08)

ax_ia = fig.add_subplot(gs[0])
ax_ia.plot(x_km, g_obs,  color="#1a1a2e", lw=1.5, label="Observed")
ax_ia.plot(x_km, g_pred, color="#f39c12", lw=1.5, ls="--", label="Predicted")
ax_ia.fill_between(x_km, g_obs, g_pred, alpha=0.25, color="#8e44ad",
                   label=f"Misfit  (RMS = {rms:.4f} mGal)")
style_anom_ax(ax_ia)
ax_ia.set_title("(b) ANN Inverted Model", pad=7)
ax_ia.legend(loc="upper left", fontsize=9)

ax_im = fig.add_subplot(gs[1])
tc2 = draw_mesh(ax_im, inv_mc, vmin=vmin, vmax=vmax)
l1, l2 = draw_horizons(ax_im)
style_model_ax(ax_im)
ax_im.set_xlabel("Distance (km)", labelpad=4)
ax_im.legend([l1, l2], ["Horizon 1", "Horizon 2"],
             loc="lower right", title="Interfaces", title_fontsize=8, fontsize=8)
for k, (pt, pp) in enumerate(zip(true_params, pred_params)):
    ax_im.annotate(
        f"ρ{k+1}: true={pt:+.0f}  →  inv={pp:+.1f} kg/m³",
        xy=(0.015, 0.97 - k*0.080), xycoords="axes fraction",
        fontsize=8.5, va="top", color="white",
        bbox=dict(boxstyle="round,pad=0.22", fc="black", alpha=0.45, ec="none"))

cax2 = fig.add_axes([0.88, 0.08, 0.022, 0.86])
cb2  = fig.colorbar(tc2, cax=cax2)
cb2.set_label("Density Contrast (kg/m³)", labelpad=8)
cb2.ax.yaxis.set_minor_locator(AutoMinorLocator(4))

plt.savefig(os.path.join(out_dir, "fig3_inverted_model.png"))
plt.close()
print("Saved fig3_inverted_model.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 – 2×2 SIDE-BY-SIDE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8.5))
gs  = gridspec.GridSpec(2, 2, height_ratios=[1, 2.5],
                        hspace=0.05, wspace=0.08,
                        left=0.07, right=0.90, top=0.93, bottom=0.08)

ax00 = fig.add_subplot(gs[0, 0])
ax00.plot(x_km, g_obs,   color="#1a1a2e", lw=1.5, label="Observed")
ax00.plot(x_km, g_clean, color="#e94560", lw=1.3, ls="--", label="Noise-free")
ax00.fill_between(x_km, g_obs, g_clean, alpha=0.13, color="#e94560")
style_anom_ax(ax00)
ax00.set_title("(a) Synthetic Forward Model", pad=6)
ax00.legend(fontsize=8)

ax10 = fig.add_subplot(gs[1, 0])
tc_s = draw_mesh(ax10, true_mc, vmin=vmin, vmax=vmax)
l1s, l2s = draw_horizons(ax10)
style_model_ax(ax10)
ax10.set_xlabel("Distance (km)")
ax10.legend([l1s, l2s], ["H1","H2"], loc="lower right",
            title="Interfaces", title_fontsize=7, fontsize=7)
for lbl, xp, yp in [("L1",37.5,-1.0),("L2",37.5,-3.5),
                     ("L3",27.5,-7.5),("L4",65.0,-7.5)]:
    ax10.text(xp, yp, lbl, ha="center", va="center",
              color="white", fontsize=8, fontweight="bold")

ax01 = fig.add_subplot(gs[0, 1])
ax01.plot(x_km, g_obs,  color="#1a1a2e", lw=1.5, label="Observed")
ax01.plot(x_km, g_pred, color="#f39c12", lw=1.5, ls="--", label="Predicted")
ax01.fill_between(x_km, g_obs, g_pred, alpha=0.25, color="#8e44ad",
                  label=f"Misfit  RMS={rms:.4f}")
style_anom_ax(ax01)
ax01.set_title("(b) ANN Inverted Model", pad=6)
ax01.legend(fontsize=8)
ax01.tick_params(labelleft=False);  ax01.set_ylabel("")

ax11 = fig.add_subplot(gs[1, 1])
tc_i = draw_mesh(ax11, inv_mc, vmin=vmin, vmax=vmax)
l1i, l2i = draw_horizons(ax11)
style_model_ax(ax11)
ax11.set_xlabel("Distance (km)")
ax11.tick_params(labelleft=False);  ax11.set_ylabel("")
ax11.legend([l1i, l2i], ["H1","H2"], loc="lower right",
            title="Interfaces", title_fontsize=7, fontsize=7)
for k, (pt, pp) in enumerate(zip(true_params, pred_params)):
    ax11.annotate(
        f"ρ{k+1}: {pt:+.0f} → {pp:+.1f} kg/m³",
        xy=(0.015, 0.97 - k*0.085), xycoords="axes fraction",
        fontsize=8, va="top", color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.40, ec="none"))

cax = fig.add_axes([0.91, 0.08, 0.018, 0.85])
cb  = fig.colorbar(tc_i, cax=cax)
cb.set_label("Density Contrast (kg/m³)", labelpad=8)
cb.ax.yaxis.set_minor_locator(AutoMinorLocator(4))

plt.savefig(os.path.join(out_dir, "fig4_comparison.png"))
plt.close()
print("Saved fig4_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 – NOISE SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
panel_colors = ["#2980b9", "#27ae60", "#e74c3c"]
panel_yl     = ["Data RMS (mGal)", "Chi-squared (χ²)", "Model RMSE (kg/m³)"]
panel_tl     = ["(a) Data Misfit", "(b) Chi-squared Fit", "(c) Model Error"]
panel_data   = [rms_values, chi2_values, model_errors]

for j, (ax_, yd, yl, tl) in enumerate(zip(axes, panel_data, panel_yl, panel_tl)):
    col = panel_colors[j]
    ax_.plot(noise_levels, yd, "o-", color=col, lw=1.8, ms=6, mew=1.4, mec="white", zorder=5)
    ax_.fill_between(noise_levels, yd, alpha=0.12, color=col)
    ax_.set_xlabel("Noise Level (%)")
    ax_.set_ylabel(yl)
    ax_.set_title(tl, pad=6)
    ax_.set_xticks(noise_levels)
    ax_.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax_.grid(ls="--", lw=0.5, color="0.85")

axes[1].axhline(1.0, color="0.45", ls=":", lw=1.4, label="χ²=1 (ideal fit)")
axes[1].legend(fontsize=8)
plt.suptitle("Noise Sensitivity Analysis", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig5_noise_sensitivity.png"))
plt.close()
print("Saved fig5_noise_sensitivity.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 – PARAMETER RECOVERY
# ─────────────────────────────────────────────────────────────────────────────
x_bar = np.arange(4)
w     = 0.35
fig, ax = plt.subplots(figsize=(7, 4.2))
b1 = ax.bar(x_bar - w/2, true_params, w, label="True",
            color="#2c3e50", edgecolor="white", lw=0.8)
b2 = ax.bar(x_bar + w/2, pred_params, w, label="ANN Predicted",
            color="#e74c3c", edgecolor="white", lw=0.8, alpha=0.87)

for rect, val in zip(b1, true_params):
    off = 4 if val >= 0 else -14
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+off,
            f"{val}", ha="center", va="bottom", fontsize=8.5)
for rect, val in zip(b2, pred_params):
    off = 4 if val >= 0 else -14
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+off,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, color="#c0392b")

ax.set_xticks(x_bar)
ax.set_xticklabels([r"$\rho_1$",r"$\rho_2$",r"$\rho_3$",r"$\rho_4$"], fontsize=12)
ax.set_ylabel("Density Contrast (kg/m³)")
ax.set_title("Parameter Recovery: True vs ANN Predicted", pad=6)
ax.legend()
ax.axhline(0, color="0.5", lw=0.7, ls="--")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.grid(axis="y", ls="--", lw=0.5, color="0.88")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig6_parameter_recovery.png"))
plt.close()
print("Saved fig6_parameter_recovery.png")




print(f"\n✓ All figures saved to: {out_dir}/")



# ─────────────────────────────────────────────────────────────
# FIGURE 7 – RESIDUAL ANALYSIS (VERY IMPORTANT)
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1,2, figsize=(10,4))

# Histogram
axes[0].hist(misfit, bins=25, color="steelblue", alpha=0.8)
axes[0].set_title("Residual Distribution")
axes[0].set_xlabel("Residual (mGal)")

# Q-Q plot
import scipy.stats as stats
stats.probplot(misfit, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "fig7_residual_analysis.png"))
plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 8 – TRUE vs PREDICTED
# ─────────────────────────────────────────────────────────────
Y_pred_test = nn.predict(X_test)

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes = axes.flatten()

for i in range(4):
    axes[i].scatter(Y_test[:,i], Y_pred_test[:,i], alpha=0.4)
    lims = [Y_test[:,i].min(), Y_test[:,i].max()]
    axes[i].plot(lims, lims, 'r--')
    axes[i].set_title(f"ρ{i+1}")
    axes[i].set_xlabel("True")
    axes[i].set_ylabel("Predicted")

plt.tight_layout()
plt.savefig(os.path.join(out_dir,"fig8_true_vs_pred.png"))
plt.close()





# ============================================================
# FULL UNCERTAINTY + DIAGNOSTIC ANALYSIS (PUBLICATION LEVEL)
# ============================================================

from sklearn.decomposition import PCA
from scipy.interpolate import griddata

print("\n─── FULL Uncertainty & Diagnostics Analysis ───")

# ============================================================
# ENSEMBLE GENERATION (POSTERIOR)
# ============================================================

n_runs = 400
noise_std = 0.05 * np.std(g_obs)

models = []
misfits = []

for _ in range(n_runs):

    g_pert = g_obs + noise_std * rng.standard_normal(len(g_obs))

    params = nn.predict([g_pert])[0]

    for i,(lo,hi) in enumerate(ranges):
        params[i] = float(np.clip(params[i], lo, hi))

    g_pred, _ = forward_from_params(params)

    misfit = np.linalg.norm(g_pert - g_pred)**2 / np.linalg.norm(g_pert)**2

    models.append(params)
    misfits.append(misfit)

models = np.array(models)
misfits = np.array(misfits)

# ============================================================
# FILTER ACCEPTABLE MODELS
# ============================================================

mask = misfits < 0.15
models_good = models[mask]
misfits_good = misfits[mask]

print("Accepted models:", len(models_good))


# ============================================================
# PARAMETER STATISTICS + CONFIDENCE INTERVALS
# ============================================================

mean_model = np.mean(models_good, axis=0)
std_model  = np.std(models_good, axis=0)

p16 = np.percentile(models_good, 16, axis=0)
p84 = np.percentile(models_good, 84, axis=0)

print("\nPosterior Statistics:")
for i in range(4):
    print(f"ρ{i+1}: {mean_model[i]:.2f} ± {std_model[i]:.2f} "
          f"[{p16[i]:.2f}, {p84[i]:.2f}]")


# ============================================================
# CORRELATION MATRIX (CLEAN)
# ============================================================

corr = np.corrcoef(models_good.T)

fig, ax = plt.subplots(figsize=(5,4))

im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

labels = [r"$\rho_1$", r"$\rho_2$", r"$\rho_3$", r"$\rho_4$"]
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(labels); ax.set_yticklabels(labels)

plt.colorbar(im, ax=ax, label="Correlation")
plt.title("Parameter Correlation Matrix")

plt.tight_layout()
plt.savefig(os.path.join(out_dir,"fig_corr_matrix.png"))
plt.close()


# ============================================================
# JOINT DISTRIBUTION (PAIRWISE SCATTER)
# ============================================================

fig, axes = plt.subplots(4,4, figsize=(8,8))

for i in range(4):
    for j in range(4):

        if i == j:
            axes[i,j].hist(models_good[:,i], bins=25, color="steelblue")
            axes[i,j].axvline(true_params[i], color="red")

        else:
            axes[i,j].scatter(models_good[:,j], models_good[:,i],
                              s=5, alpha=0.3, color="black")

        if i == 3:
            axes[i,j].set_xlabel(labels[j])
        if j == 0:
            axes[i,j].set_ylabel(labels[i])

plt.tight_layout()
plt.savefig(os.path.join(out_dir,"fig_joint_pdf.png"))
plt.close()


# ============================================================
# MISFIT vs PARAMETERS (SENSITIVITY)
# ============================================================

fig, axes = plt.subplots(1,4, figsize=(14,3))

for i in range(4):

    axes[i].scatter(models_good[:,i], misfits_good,
                    s=10, alpha=0.5)

    axes[i].set_xlabel(labels[i])
    axes[i].set_ylabel("Misfit")

plt.tight_layout()
plt.savefig(os.path.join(out_dir,"fig_misfit_vs_params.png"))
plt.close()


# ============================================================
# PCA (FIXED BASIS)
# ============================================================

pca = PCA(n_components=2)
Xp = pca.fit_transform(models_good)

best_idx = np.argmin(misfits_good)
best_model = models_good[best_idx]

best_pca = pca.transform(best_model.reshape(1,-1))
true_pca = pca.transform(np.array(true_params).reshape(1,-1))


# ============================================================
# PCA COST FUNCTION (SCATTER)
# ============================================================

fig, ax = plt.subplots(figsize=(6,5))

sc = ax.scatter(
    Xp[:,0], Xp[:,1],
    c=misfits_good*100,
    cmap="viridis",
    s=30
)

plt.colorbar(sc, label="Misfit (%)")

ax.scatter(best_pca[:,0], best_pca[:,1],
           c="red", s=120, marker="*", label="Best")

ax.scatter(true_pca[:,0], true_pca[:,1],
           c="blue", s=100, marker="D", label="True")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

ax.set_title("PCA Cost Function Topography")

ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir,"fig_pca_scatter_final.png"))
plt.close()


# ============================================================
# PCA CONTOUR (SMOOTH COST LANDSCAPE)
# ============================================================

x = Xp[:,0]; y = Xp[:,1]
z = misfits_good * 100

xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)

XI, YI = np.meshgrid(xi, yi)
ZI = griddata((x,y), z, (XI,YI), method="linear")

fig, ax = plt.subplots(figsize=(6,5))

cf = ax.contourf(XI, YI, ZI, levels=30, cmap="viridis")
ax.contour(XI, YI, ZI, levels=15, colors="k", linewidths=0.4)

ax.scatter(best_pca[:,0], best_pca[:,1],
           c="red", marker="*", s=150, label="Best")

ax.scatter(true_pca[:,0], true_pca[:,1],
           c="blue", marker="D", s=120, label="True")

plt.colorbar(cf, label="Misfit (%)")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

ax.set_title("PCA Cost Function (Contour)")

ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir,"fig_pca_contour_final.png"))
plt.close()


print("✓ FULL uncertainty + diagnostics completed")





# ============================================================
# MULTI-NOISE TRAINING (PUBLICATION LEVEL)
# ============================================================

noise_levels_train = [0, 2, 5, 10, 20]
loss_curves = {}

print("\n=== Multi-noise ANN Training ===")

for nl in noise_levels_train:

    print(f"\nTraining for {nl}% noise...")

    # --------------------------------------------------------
    # GENERATE DATASET WITH SPECIFIC NOISE
    # --------------------------------------------------------
    X_data = np.zeros((n_samples, len(pnts)))
    Y_data = np.zeros((n_samples, 4))

    for i in range(n_samples):
        params = [float(rng.uniform(lo, hi)) for lo, hi in ranges]
        g, _   = forward_from_params(params)

        noise_std = (nl/100.0)*np.std(g) if nl > 0 else 0.001*np.std(g)
        noise = noise_std * rng.standard_normal(g.shape)

        X_data[i] = g + noise
        Y_data[i] = params

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_data, Y_data, test_size=0.2, random_state=42
    )

    # --------------------------------------------------------
    # TRAIN ANN
    # --------------------------------------------------------
    nn_local = MLPRegressor(
        hidden_layer_sizes=(256,128,64),
        activation="relu",
        max_iter=5000,
        tol=1e-4,
        random_state=42,
        verbose=False
    )

    nn_local.fit(X_train, Y_train)

    loss_curves[nl] = nn_local.loss_curve_

    print(f"Final loss ({nl}%): {nn_local.loss_curve_[-1]:.4f}")






# ============================================================
# FIGURE 1 – TRAINING CONVERGENCE (MULTI-NOISE)
# ============================================================

fig, ax = plt.subplots(figsize=(7,4.5))

colors = ["#2c3e50","#2980b9","#27ae60","#f39c12","#e74c3c"]

for i, nl in enumerate(noise_levels_train):

    loss = loss_curves[nl]
    epochs = np.arange(1, len(loss)+1)

    ax.plot(
        epochs,
        loss,
        lw=2,
        color=colors[i],
        label=f"{nl}% noise"
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")

ax.set_yscale("log")

ax.set_title("ANN Training Convergence under Different Noise Levels")

ax.legend(title="Noise Level")

ax.grid(which="major", ls="--", lw=0.5, color="0.8")
ax.grid(which="minor", ls=":", lw=0.3, color="0.9")

ax.xaxis.set_minor_locator(AutoMinorLocator(5))

plt.tight_layout()

plt.savefig(os.path.join(out_dir, "fig1_training_multi_noise.png"))

plt.close()

print("✓ Saved multi-noise training convergence")





import pygimli as pg
import matplotlib.pyplot as plt
import os

# ===== Output folder =====
out_dir = "AGM_ANN_pub"
os.makedirs(out_dir, exist_ok=True)

# ===== Assign densities to regions (same order as markers) =====
# markers: 1=top, 2=middle, 3=bottom_left, 4=bottom_right
rho = true_params  # [rho1, rho2, rho3, rho4]

model = []
for cell in mesh_shared.cells():
    marker = cell.marker()
    if marker == 1:
        model.append(rho[0])
    elif marker == 2:
        model.append(rho[1])
    elif marker == 3:
        model.append(rho[2])
    elif marker == 4:
        model.append(rho[3])
    else:
        model.append(0)

model = pg.Vector(model)

# ===== Plot =====
fig, ax = plt.subplots(figsize=(10,6))

pg.show(mesh_shared,
        data=model,
        cmap="viridis",
        ax=ax,
        showMesh=True)   # ← THIS enables mesh lines

# Labels
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Depth (m)")
ax.set_title("Synthetic Geological Model with Mesh")


plt.tight_layout()
plt.savefig(os.path.join(out_dir, "synthetic_mesh_model.png"), dpi=600)
plt.close()

print("✓ Mesh-based geological model saved as PNG")



