# ConNeCT: Weakly-Supervised Corneal Confocal Microscopy Image Inpainting Network Based on Diffusion Model


## Example
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from utils.inpaint import inpaint
from utils.load_model import load_model

models = load_model()

inpaint(
    './data/ori.png',
    './data/anno.png',
    './data/rst.png',
    models=models
)
```

## Citation

```text
Qiao, Qincheng, and Xinguo Hou. “ConNeCT: weakly supervised corneal confocal microscopy image inpainting network based on a diffusion model.” Biomedical optics express vol. 16,7 2615-2630. 9 Jun. 2025, doi:10.1364/BOE.562924
```

