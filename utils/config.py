class Config:
    def __init__(self):
        self._config = {
            'attention_resolutions': '16,8',
            'class_cond': False,
            'diffusion_steps': 1000,
            'learn_sigma': False,
            'noise_schedule': 'linear',
            'num_channels': 32,
            'num_head_channels': -1,
            'num_heads': 4,
            'num_res_blocks': 1,
            'resblock_updown': False,
            'use_fp16': False,
            'use_scale_shift_norm': True,
            'classifier_scale': 1,
            'lr_kernel_n_std': 2,
            'num_samples': 100,
            'show_progress': True,
            'timestep_respacing': '',
            'use_kl': False,
            'predict_xstart': False,
            'rescale_timesteps': False,
            'rescale_learned_sigmas': False,
            'num_heads_upsample': -1,
            'channel_mult': '',
            'dropout': 0.0,
            'use_checkpoint': False,
            'use_new_attention_order': False,
            'clip_denoised': True,
            'use_ddim': False,
            'image_size': 384,
            'name': 'test_inet256_ev2li',
            'inpa_inj_sched_prev': True,
            'n_jobs': 25,
            'print_estimated_vars': True,
            'inpa_inj_sched_prev_cumnoise': False,
            'resampling': {
                'T': 250,
                'N': 10,
                'M': 10
            }
        }

    def __getattr__(self, name):
        try:
            return self._config[name]
        except KeyError:
            return None

    def __getitem__(self, key):
        return self._config[key]

    def __contains__(self, key):
        return key in self._config

    def keys(self):
        return self._config.keys()

    def items(self):
        return self._config.items()

    def values(self):
        return self._config.values()


config = Config()
