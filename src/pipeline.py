import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from dataloader import get_loader
from model import Generator, Discriminator
from train import train_epoch
import utils

def train_pipeline(src_data_path, trg_data_path, experement_name, 
                   config_path='./config.yaml', device='cpu', scheduler='Linear', 
                   warm_lr=False, num_epoch=None, lr=None, img_size=None, 
                   batch_size=None, save_path=None, load_path=None, save_period=1):
    
    config = utils.load_config(config_path=config_path)
    
    IMG_CHANNELS = config['MODEL']['IMG_CHANNELS']
    NUM_HID_CHANNELS = config['MODEL']['NUM_HID_CHANNELS']
    NUM_RESIDUALS = config['MODEL']['NUM_RESIDUALS']
    IMG_SIZE = img_size if img_size else config['DATASET']['IMG_SIZE']
    BATCH_SIZE = batch_size if batch_size else config['DATASET']['BATCH_SIZE']
    STATS = config['DATASET']['STATS']
    NUM_EPOCHS = num_epoch if num_epoch else config['TRAINING']['NUM_EPOCHS']
    MIN_LR = config['TRAINING']['MIN_LR']
    LR = lr if lr else config['TRAINING']['LR']
    LR_RED_FACTOR = config['TRAINING']['LR_RED_FACTOR']
    START_RED_LR = config['TRAINING']['START_RED_LR']
    BETAS = config['TRAINING']['BETAS']
    LAMBDA_CYCLE = config['TRAINING']['LAMBDA_CYCLE']
    SAVE_MODEL = True if save_path else False
    LOAD_CHECKPOINT = True if load_path else False
    DEVICE = device
    EPOCH = 0
    
    gen_trg = Generator(img_ch=IMG_CHANNELS, 
                        num_hid_channels=NUM_HID_CHANNELS, 
                        num_residuals=NUM_RESIDUALS).to(DEVICE)
    gen_src = Generator(img_ch=IMG_CHANNELS, 
                        num_hid_channels=NUM_HID_CHANNELS, 
                        num_residuals=NUM_RESIDUALS).to(DEVICE)
    dis_trg = Discriminator(img_ch=IMG_CHANNELS, 
                            num_hid_channels=NUM_HID_CHANNELS).to(DEVICE)
    dis_src = Discriminator(img_ch=IMG_CHANNELS, 
                            num_hid_channels=NUM_HID_CHANNELS).to(DEVICE)
    
    opt_gen = optim.Adam(params=list(gen_trg.parameters()) + list(gen_src.parameters()), 
                         lr=LR, betas=BETAS)
    opt_dis = optim.Adam(params=list(dis_trg.parameters()) + list(dis_src.parameters()), 
                         lr=LR, betas=BETAS)
    
    if scheduler == 'Linear':
        def lr_lambda(epoch):
            if epoch > START_RED_LR:
                return LR_RED_FACTOR
            else:
                return 1
        scheduler_gen = optim.lr_scheduler.MultiplicativeLR(optimizer=opt_gen, 
                                                            lr_lambda=lr_lambda)
        scheduler_dis = optim.lr_scheduler.MultiplicativeLR(optimizer=opt_dis, 
                                                            lr_lambda=lr_lambda)
    elif scheduler == 'Cyclic':
        scheduler_gen = optim.lr_scheduler.CyclicLR(optimizer=opt_gen, 
                                                    base_lr=MIN_LR, 
                                                    max_lr=LR, 
                                                    step_size_up=5, 
                                                    mode="triangular")
        scheduler_dis = optim.lr_scheduler.CyclicLR(optimizer=opt_dis, 
                                                    base_lr=MIN_LR, 
                                                    max_lr=LR, 
                                                    step_size_up=5, 
                                                    mode="triangular")
    
        
    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    gen_scal = torch.cuda.amp.GradScaler()
    dis_scal = torch.cuda.amp.GradScaler()
    
    loader = get_loader(source_path=src_data_path, target_path=trg_data_path, 
                        img_size=IMG_SIZE, stats=STATS, batch_size=BATCH_SIZE)
    
    if LOAD_CHECKPOINT:
        EPOCH += utils.load_checkpoint(chekpoints_path=load_path, device=DEVICE, 
                                       gen_trg=gen_trg, gen_src=gen_src, 
                                       dis_trg=dis_trg, dis_src=dis_src, 
                                       opt_dis=opt_dis, opt_gen=opt_gen, 
                                       lr_opt_gen=warm_lr, lr_opt_dis=warm_lr)
        
    writer = SummaryWriter(comment=experement_name)
    
    for epoch in tqdm(range(NUM_EPOCHS), leave=True):
        (loss_dis_ep, loss_gen_ep,
         real_score_ep, fake_score_ep, 
         val_trg_img, val_src_img) = train_epoch(gen_trg=gen_trg, 
                                                 gen_src=gen_src, 
                                                 dis_trg=dis_trg, 
                                                 dis_src=dis_src, 
                                                 loader=loader, 
                                                 opt_gen=opt_gen, 
                                                 opt_dis=opt_dis, 
                                                 l1=L1, mse=mse, 
                                                 dis_scal=dis_scal, 
                                                 gen_scal=gen_scal, 
                                                 lambda_cycle=LAMBDA_CYCLE, 
                                                 device=DEVICE, 
                                                 scheduler_dis=scheduler_dis, 
                                                 scheduler_gen=scheduler_gen)
        
        if SAVE_MODEL and (((epoch+1) % save_period == 0) or (epoch+1) == NUM_EPOCHS):
            utils.save_checkpoint(gen_trg=gen_trg, gen_src=gen_src, 
                                  dis_trg=dis_trg, dis_src=dis_src, 
                                  opt_dis=opt_dis, opt_gen=opt_gen, 
                                  chekpoints_path=save_path, 
                                  epoch=EPOCH + epoch + 1)
            
        trg_img = utils.img_grid(img_tensors=val_trg_img, stats=STATS)
        src_img = utils.img_grid(img_tensors=val_src_img, stats=STATS)
        
        writer.add_scalar('Loss/Discriminator', loss_dis_ep, epoch)
        writer.add_scalar('Loss/Generator', loss_gen_ep, epoch)
        writer.add_scalar('Real_score', real_score_ep, epoch)
        writer.add_scalar('Fake_score', fake_score_ep, epoch)
        writer.add_image('Target_image', trg_img, epoch)
        writer.add_image('Source_image', src_img, epoch)
        
        utils.show_images(trg_img)
        utils.show_images(src_img)
        
        print(f'Epoch: {epoch+1}')
        print(f'Discriminator loss: {loss_dis_ep}')
        print(f'Generator loss: {loss_gen_ep}')
        print(f'Real score: {real_score_ep}')
        print(f'Fake score: {fake_score_ep}')
    
    writer.flush()
    writer.close()