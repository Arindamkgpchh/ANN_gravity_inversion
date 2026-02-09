import torch
import matplotlib.pyplot as plt
import pygimli as pg




def invert(net, A, mesh, norm_g, assemble_fn, x, g, args, device):
net.eval()
with torch.no_grad():
g_t = torch.tensor(g, dtype=torch.float32, device=device)[None,:]
r_free = net(norm_g(g_t))
rho = assemble_fn(r_free)[0].cpu().numpy()


g_pred = A @ rho


fig, ax = plt.subplots(2,1, figsize=(10,7))
ax[0].plot(x/1000, g, 'k.', label='Observed')
ax[0].plot(x/1000, g_pred, 'r-', label='Predicted')
ax[0].legend(); ax[0].grid()


pg.show(mesh, rho, ax=ax[1], cMap='seismic', showMesh=False)
plt.show()