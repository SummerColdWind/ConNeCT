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
