import yaml
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def denorm(img_tensors, stats):
    return img_tensors * stats[1][0] + stats[0][0]

def img_grid(img_tensors, stats, nmax=16, nrow=8):
    return make_grid(denorm(img_tensors[:nmax], stats), nrow=nrow)

def show_images(img_tensors):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img_tensors.permute(1, 2, 0))
    
def save_checkpoint():
    pass

def load_checkpoint():
    pass
    
