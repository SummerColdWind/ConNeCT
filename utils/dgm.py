import torch

from . import device


def get_conf_fn(total_timesteps, binary):
    def cond_fn(x, t, y=None, gt=None, **kwargs):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            masked_pixels = x_in * binary
            loss = -torch.mean(masked_pixels ** 2)
            grad = torch.autograd.grad(loss, x_in)[0]
            t_norm = t.float() / total_timesteps
            guidance_strength = 2.0 * (1 - t_norm)

            grad = grad * guidance_strength
            grad = grad.to(device)
            return grad

    return cond_fn
