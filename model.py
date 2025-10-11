# model.py
import torch
import torch.nn as nn
import math

# ---------- Positional Encoding ----------
class SinusEncoding(nn.Module):
    def __init__(self, num_freqs: int, input_dims: int = 3):
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dims = input_dims
        self.output_dims = input_dims * (2 * num_freqs + 1)

        self.register_buffer("freqs", 2.0 ** torch.arange(num_freqs, dtype=torch.float32))
        
        
    def forward(self, x: torch.Tensor):

        freqs = self.freqs.to(x.device)
        scaled = x.unsqueeze(-1) * freqs * torch.pi
        sinusoids = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        sinusoids = sinusoids.view(*x.shape[:-1], -1)

        return torch.cat([x, sinusoids], dim=-1)

class GaussianFourierEncoding(nn.Module):
    def __init__(self, num_features: int = 256, sigma: float = 10.0, input_dims: int = 3):
        super().__init__()
        self.input_dims = input_dims
        self.num_features = num_features
        
        self.sigma = sigma
        self.output_dims = 2 * num_features

        B = torch.randn((input_dims, num_features)) * sigma
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor):
        B = self.B.to(x.device)
        x = x / torch.max(torch.abs(x), dim=-1, keepdim=True)[0].clamp(min=1e-6)
        x_proj = 2 * math.pi * x @ B  

        encoded = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return encoded

# ---------- NeRF Model ----------
class NeRF(nn.Module):
    def __init__(self,
                 pos_input_size,
                 pos_dir_size,
                 n_neurons=256) :
        super().__init__()
        self.linear=nn.Linear(n_neurons,n_neurons)

        self.mlp_1 = nn.Sequential(
            nn.Linear(pos_input_size,n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons,n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons,n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons,n_neurons),
            nn.ReLU()
        )

        # With skip connexion
        self.mlp_2 = nn.Sequential(
            nn.Linear(pos_input_size + n_neurons,n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons,n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons,n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons,n_neurons),
            nn.ReLU()
        )

        self.decode_sigma=nn.Sequential(
            nn.Linear(n_neurons,1),
            nn.ReLU(),
        )

        self.decode_rgb = nn.Sequential(
            nn.Linear(pos_dir_size + n_neurons, n_neurons // 2),
            nn.ReLU(),
            nn.Linear(n_neurons // 2, 3),
            nn.Sigmoid() 
        )

    def forward(self, x: torch.Tensor, dir: torch.Tensor):
        
        x_mlp1 = self.mlp_1(x)
        x_mlp2 = self.mlp_2(torch.cat([x_mlp1, x], dim=-1))

        sigma = self.decode_sigma(x_mlp2)

        x_linear = self.linear(x_mlp2)

        rgb = self.decode_rgb(torch.cat([x_linear, dir], dim=-1))

        return rgb, sigma

if __name__ == "__main__":

    L_xyz, L_dir = 10, 4
    encoder_input = SinusEncoding(num_freqs=L_xyz)
    encoder_dir = SinusEncoding(num_freqs=L_dir)
    
    x = torch.randn(2, 3)
    d = torch.randn(2, 3)

    x_embed = encoder_input(x)
    d_embed = encoder_dir(d)

    model = NeRF(pos_input_size=x_embed.shape[-1],
                 pos_dir_size=d_embed.shape[-1],
                 n_neurons=256)

    rgb, sigma = model(x_embed, d_embed)
    print("RGB:", rgb.shape, "Sigma:", sigma.shape)

    encoder_input = GaussianFourierEncoding(input_dims=3, num_features=256, sigma=10.0)
    encoder_dir = GaussianFourierEncoding(input_dims=3, num_features=128, sigma=4.0)

    x = torch.randn(2, 3)
    d = torch.randn(2, 3)

    x_embed = encoder_input(x)
    d_embed = encoder_dir(d)

    print("Encoded shapes:", x_embed.shape, d_embed.shape)



