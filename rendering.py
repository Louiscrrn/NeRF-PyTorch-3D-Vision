import torch
from sampling import sample, dep_to_pos
from rays_utils import get_rays
from tqdm import tqdm
from sampling import sample

def volume_render_pass(network_fn, rays_o, rays_d, z_vals, posenc, device='cuda'):
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    pts_flat = posenc(pts.reshape(-1, 3))
    raw = network_fn(pts_flat).reshape(*pts.shape[:-1], 4)

    rgb = torch.sigmoid(raw[..., :3])
    sigma = torch.relu(raw[..., 3])
    return volume_render(rgb, sigma, z_vals)

def volume_render(rgb, sigma, z_vals, white_bkgd=True):

    deltas = z_vals[..., 1:] - z_vals[..., :-1]
    deltas = torch.cat([deltas, torch.full_like(deltas[..., :1], torch.finfo(z_vals.dtype).max)], dim=-1)

    alpha = 1. - torch.exp(-sigma * deltas)
    T = cumprod(1. - alpha + 1e-10)
    weights = alpha * T

    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map.unsqueeze(-1))

    return rgb_map, depth_map, acc_map, weights

def cumprod(tensor):
    ones = torch.ones_like(tensor[..., :1])
    return torch.cumprod(torch.cat([ones, tensor[..., :-1]], dim=-1), dim=-1)


def render_batch_of_rays( model, pos_enc_input, pos_enc_dir,
                        rays_o, rays_d, near, far, N_samples,
                    white_bkgd=True, device="cpu", chunk_size=4096
):

    z_vals = sample(rays_o, rays_d, near, far, N_samples, perturb=True, device=device)

    pts = dep_to_pos(rays_o, rays_d, z_vals)

    pts_flat = pts.reshape(-1, 3)
    dirs_flat = rays_d.unsqueeze(1).expand_as(pts).reshape(-1, 3)
    encoded_input = pos_enc_input(pts_flat)
    encoded_dir = pos_enc_dir(dirs_flat)

    rgb_list, sigma_list = [], []
    for i in range(0, encoded_input.shape[0], chunk_size):
        rgb, sigma = model(encoded_input[i:i+chunk_size], encoded_dir[i:i+chunk_size])
        rgb_list.append(rgb)
        sigma_list.append(sigma)

    rgb = torch.cat(rgb_list, dim=0).reshape(rays_o.shape[0], N_samples, 3)
    sigma = torch.cat(sigma_list, dim=0).reshape(rays_o.shape[0], N_samples)
    return volume_render(rgb, sigma, z_vals, white_bkgd)


def render_full_image(
    model, encoder_input, encoder_dir, H: int, W: int, focal: float, c2w: torch.Tensor,
    near: float, far: float, n_samples: int, white_bkgd: bool = False, device: str = "cpu", chunk_size: int = 4096,
):
    """
    Render a full image (all rays) from a trained NeRF model.
    """
    rays_o, rays_d = get_rays(H, W, focal, c2w)
    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)

    all_rgb_chunks = []

    for i in range(0, rays_o.shape[0], chunk_size):
        rays_o_chunk = rays_o[i:i + chunk_size]
        rays_d_chunk = rays_d[i:i + chunk_size]

        rgb_map, depth_map, acc_map, weights = render_batch_of_rays(
            model=model,
            pos_enc_input=encoder_input,
            pos_enc_dir=encoder_dir,
            rays_o=rays_o_chunk,
            rays_d=rays_d_chunk,
            near=near,
            far=far,
            N_samples=n_samples,
            white_bkgd=white_bkgd,
            device=device,
            chunk_size=chunk_size,
        )

        all_rgb_chunks.append(rgb_map)

    rendered_img = torch.cat(all_rgb_chunks, dim=0).reshape(H, W, 3)
    return rendered_img


if __name__ == "__main__":
    from model import NeRF, PositionalEncoding
    from dataset import LegoDataset
    import imageio

    dataset = LegoDataset("data/tiny_nerf_data.npz", split="val")
    elem = dataset[0]

    encoder_input = PositionalEncoding(num_freqs=6, input_dims=3).to("cpu")
    encoder_dir = PositionalEncoding(num_freqs=4, input_dims=3).to("cpu")
    model = NeRF(pos_input_size=encoder_input.output_dims, pos_dir_size=encoder_dir.output_dims).to("cpu")

    rendered = render_full_image(
        model=model,
        encoder_input=encoder_input,
        encoder_dir=encoder_dir,
        H=dataset.H,
        W=dataset.W,
        focal=dataset.focal,
        c2w=elem["c2w"],
        near=2.0,
        far=6.0,
        n_samples=64,
        white_bkgd=False,
        device="cpu",
    )

    imageio.imwrite("render_val.png", (rendered.detach().numpy() * 255 * 255).astype("uint8"))
