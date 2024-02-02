import yaml
import torch
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
    plt.show()
    
def plot_train_process(loss_dis, loss_gen, real_score, fake_score):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].set_title('Loss')
    axes[0].plot(loss_dis, label='discriminator')
    axes[0].plot(loss_gen, label='generator')
    axes[0].legend()

    axes[1].set_title('Score')
    axes[1].plot(real_score, label='Real score')
    axes[1].plot(fake_score, label='Fake score')
    axes[1].legend()
    
    plt.show()
    
def save_checkpoint(gen_trg, gen_src, dis_trg, dis_src,
                    opt_dis, opt_gen, chekpoints_path, epoch):
    path = f'{chekpoints_path}/models_{epoch}_epoch'
    torch.save({'generator_trg_state': gen_trg.state_dict(), 
                'generator_src_state': gen_src.state_dict(),
                'discriminator_trg_state': dis_trg.state_dict(),
                'discriminator_src_state': dis_src.state_dict(),
                'optimizer_dis_state': opt_dis.state_dict(),
                'optimizer_gen_state': opt_gen.state_dict(),
                'lr_opt_dis': opt_dis.param_groups[0]["lr"],
                'lr_opt_gen': opt_gen.param_groups[0]["lr"],
                'epoch': epoch}, path)
    

def load_checkpoint(chekpoints_path, device, gen_trg=None, gen_src=None, 
                    dis_trg=None, dis_src=None, opt_dis=None, 
                    opt_gen=None, lr_opt_dis=False, lr_opt_gen=False):
    chekpoint = torch.load(chekpoints_path, map_location=torch.device(device))
    
    if gen_trg:
        gen_trg.load_state_dict(chekpoint['generator_trg_state'])
    if gen_src:
        gen_src.load_state_dict(chekpoint['generator_src_state'])
    if dis_trg:
        dis_trg.load_state_dict(chekpoint['discriminator_trg_state'])
    if dis_src:
        dis_src.load_state_dict(chekpoint['discriminator_src_state'])
    if opt_dis:
        opt_dis.load_state_dict(chekpoint['optimizer_dis_state'])
    if opt_gen:
        opt_gen.load_state_dict(chekpoint['optimizer_gen_state'])
    if lr_opt_dis:
        opt_dis.param_groups[0]["lr"] = chekpoint['lr_opt_dis']
    if lr_opt_gen:
        opt_gen.param_groups[0]["lr"] = chekpoint['lr_opt_gen']
    
    return chekpoint['epoch']