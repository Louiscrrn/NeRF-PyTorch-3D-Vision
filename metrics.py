import torch

def psnr_metric(img_true, img_pred):
    mse = torch.mean((img_true - img_pred) ** 2)
    if mse < 1e-10:
        return float('inf')
    return -10.0 * torch.log10(mse)