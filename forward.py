import numpy as np
from pygimli.physics.gravimetry import GravityModelling2D




def build_operator(mesh, xstations):
pnts = np.column_stack([xstations, np.zeros_like(xstations)])
fop = GravityModelling2D(mesh=mesh, points=pnts)
M = mesh.cellCount()
N = len(xstations)
A = np.zeros((N, M))
unit = np.zeros(M)
for j in range(M):
unit[:] = 0.0
unit[j] = 1.0
A[:, j] = fop.response(unit)
return A