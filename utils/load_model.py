from guided_diffusion import dist_util
from guided_diffusion.script_util import create_model_and_diffusion, select_args, model_and_diffusion_defaults

import os

from . import device
from .config import config

model_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        'model.pt'
    )
)

def load_model():
    model, diffusion = create_model_and_diffusion(
        **select_args(config, model_and_diffusion_defaults().keys()), conf=config
    )
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(device)
    model.eval()

    return model, diffusion
