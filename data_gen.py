import numpy as np


def smooth_x(vals, xs, corr_len=6000.0):
w = np.exp(-0.5*((xs[:,None]-xs[None,:])/corr_len)**2)
w /= w.sum(axis=1, keepdims=True)
return w @ vals




def random_layered_field(M, idx1, idx2, idx3, cx, args):
rho = np.full(M, args.rho3_fixed)
if len(idx1):
v1 = np.random.uniform(args.rho1_min, args.rho1_max, len(idx1))
rho[idx1] = smooth_x(v1, cx[idx1])
if len(idx2):
v2 = np.random.uniform(args.rho2_min, args.rho2_max, len(idx2))
rho[idx2] = smooth_x(v2, cx[idx2])
rho[idx3] = args.rho3_fixed
return rho




def sample_batch(batch, A, Mfree, idx_free, gen_fun, noise_std):
R, G = [], []
for _ in range(batch):
rho = gen_fun()
g = A @ rho
if noise_std > 0:
g += np.random.normal(0, noise_std, size=len(g))
R.append(rho[idx_free])
G.append(g)
return np.stack(G), np.stack(R)