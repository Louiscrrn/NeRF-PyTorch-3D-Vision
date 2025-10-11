import torch


def get_rays(H, W, focal, c2w):
        
        i, j = torch.meshgrid(
            torch.linspace(0, W - 1, W),
            torch.linspace(0, H - 1, H),
            indexing='xy'
        )
        dirs = torch.stack([(i - W * 0.5) / focal,
                            -(j - H * 0.5) / focal,
                            -torch.ones_like(i)], dim=-1)

        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  

        rays_o = c2w[:3, 3].expand(rays_d.shape)
        
        return rays_o, rays_d