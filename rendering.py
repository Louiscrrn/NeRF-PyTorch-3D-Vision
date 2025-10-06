import torch
import torch.nn.functional as F
from model import posenc

def get_rays(H, W, focal, c2w, device='cuda'):
    if isinstance(H, torch.Tensor):
        H = int(H.item())
    if isinstance(W, torch.Tensor):
        W = int(W.item())
    if isinstance(focal, torch.Tensor):
        focal = float(focal.item())

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='xy'
    )
    dirs = torch.stack([(i - W * 0.5) / focal,
                        -(j - H * 0.5) / focal,
                        -torch.ones_like(i)], dim=-1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, -1].expand_as(rays_d)
    return rays_o, rays_d


def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, L_embed=6, rand=False, device='cuda'):
    rays_o, rays_d = rays_o.float(), rays_d.float()
    z_vals = torch.linspace(near, far, N_samples, device=device).float()
    if rand:
        z_vals += torch.rand_like(z_vals) * (far - near) / N_samples

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[None, None, :, None]
    pts_flat = pts.reshape(-1, 3)
    pts_flat = posenc(pts_flat, L_embed=L_embed)

    raw = []
    chunk = 1024 * 32
    for i in range(0, pts_flat.shape[0], chunk):
        raw.append(network_fn(pts_flat[i:i + chunk]))
    raw = torch.cat(raw, dim=0)
    raw = raw.reshape(list(pts.shape[:-1]) + [4])

    sigma_a = F.relu(raw[..., 3])
    rgb = torch.sigmoid(raw[..., :3])
    dists = torch.cat([
        z_vals[..., 1:] - z_vals[..., :-1],
        torch.ones_like(z_vals[..., :1]) * 1e10
    ], dim=-1)

    alpha = 1. - torch.exp(-sigma_a * dists)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = alpha * T

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map
