from .dgm import get_conf_fn
from .parse_input import parse_input, MASK_MODE
from .save import save_result
from .config import config
from . import device


def inpaint(
        ori_path, mask_path, save_path,
        mask_mode: MASK_MODE = 'green',
        progress_bar=True,
        *,
        models
):
    model, diffusion = models

    def model_fn(x, t, y=None, gt=None, **kwargs):
        return model(x, t, None, gt=gt)

    ori, mask, binary = parse_input(ori_path, mask_path, mask_mode)
    model_kwargs = {'gt_keep_mask': mask, "gt": ori}

    cond_fn = get_conf_fn(config.resampling['T'], binary)

    result = diffusion.p_sample_loop(
        model_fn,
        (1, 1, 384, 384),
        clip_denoised=config.clip_denoised,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
        device=device,
        progress=progress_bar,
        return_all=True,
        conf=config
    )

    save_result(result, save_path)
