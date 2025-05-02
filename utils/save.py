import torch
import aiccm


def save_result(result, path):
    sample = result['sample']
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    sample = sample[0]
    aiccm.save_image(sample, path)



