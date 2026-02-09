import torch
import torch.nn.functional as F
from utils import format_seconds




def train(net, opt, sampler, norm_g, args, device):
hist = []
for ep in range(args.epochs):
net.train()
total = 0
for _ in range(args.n_samples//args.batch_size):
g, r = sampler(args.batch_size)
g = torch.tensor(g, dtype=torch.float32, device=device)
r = torch.tensor(r, dtype=torch.float32, device=device)


pred = net(norm_g(g))
loss = F.mse_loss(pred, r)


opt.zero_grad()
loss.backward()
opt.step()
total += loss.item()


hist.append(total)
print(f"Epoch {ep+1}: loss={total}")
return hist