import numpy as np


def interp_fun(pts):
pts = np.array(pts, float)
xs, zs = pts[:,0], pts[:,1]
srt = np.argsort(xs)
xs, zs = xs[srt], zs[srt]
return lambda xq: np.interp(xq, xs, zs, left=zs[0], right=zs[-1])




def build_layer_masks(cx, cz_neg, z1, z2):
z1c = z1(cx)
z2c = z2(cx)


mask1 = (cz_neg <= 0.0) & (cz_neg >= z1c)
mask2 = (cz_neg < z1c) & (cz_neg >= z2c)
mask3 = (cz_neg < z2c)


idx1 = np.where(mask1)[0]
idx2 = np.where(mask2)[0]
idx3 = np.where(mask3)[0]


idx_free = np.concatenate([idx1, idx2])
return idx1, idx2, idx3, idx_free