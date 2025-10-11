import torch

def sample(ray_origins, ray_directions, near, far, n_coarse, perturb=True, device='cuda'):
    """
    Sample uniformly depths between near and far along rays.
    """
    n_rays = ray_origins.shape[0]
    z_starts = torch.linspace(near, far, n_coarse + 1, device=device)[:-1]  # shape (n_coarse,)
    depths = z_starts.unsqueeze(0).expand(n_rays, n_coarse).clone()
    gap = (far - near) / n_coarse

    if perturb:
        offsets = torch.rand_like(depths) * gap
        depths += offsets
    else:
        depths += gap * 0.5

    return depths

def dep_to_pos(ray_origins, ray_directions, depths):
    """
    Convert sampled depth in 3D positions.
    """
    return ray_origins.unsqueeze(1) + depths.unsqueeze(2) * ray_directions.unsqueeze(1)


if __name__ == "__main__":
    torch.manual_seed(0)
    N_rays = 4
    rays_o = torch.zeros((N_rays, 3))
    rays_d = torch.tensor([[0, 0, -1]]).expand(N_rays, 3)
    near, far = 2.0, 6.0

    # Test coarse sampling
    z_coarse = sample(rays_o, rays_d, near, far, 8, perturb=True, device='cpu')
    print("Coarse z_vals:\n", z_coarse)

