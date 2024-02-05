import numpy as np
import torch
from tqdm.auto import tqdm

def train_epoch(gen_trg, gen_src, dis_trg, dis_src, 
                loader, opt_gen, opt_dis, l1, mse, 
                lambda_cycle, idt_coef, device, 
                scheduler_dis, scheduler_gen, 
                buffer_trg, buffer_src):
    
    gen_trg.train()
    gen_src.train()
    dis_trg.train()
    dis_src.train()
    
    loss_dis_ep = []
    loss_gen_ep = []
    real_score_ep = []
    fake_score_ep = []
    val_idx = np.random.randint(len(loader))
    
    for idx, (src, trg) in enumerate(tqdm(loader, leave=False)):
        if idx == val_idx:
            val_src = src.clone()
            val_trg = trg.clone()
        src = src.to(device)
        trg = trg.to(device)
        
        # Train generators
        #fake_img
        fake_trg = gen_trg(src)
        fake_src = gen_src(trg)
        #recon_img
        cycle_trg = gen_trg(fake_src)
        cycle_src = gen_src(fake_trg)
        #ident_img
        idt_trg = gen_trg(trg)
        idt_src = gen_src(src)
        
        #Adversarial loss
        trg_fake_pred = dis_trg(fake_trg)
        src_fake_pred = dis_src(fake_src)
        adv_trg_loss = mse(trg_fake_pred, torch.ones_like(trg_fake_pred))
        adv_src_loss = mse(src_fake_pred, torch.ones_like(src_fake_pred))
        total_adv_loss = adv_trg_loss + adv_src_loss
        
        #Cycle loss
        cycle_trg_loss = l1(cycle_trg, trg)
        cycle_src_loss = l1(cycle_src, src)
        total_cycle_loss = cycle_trg_loss + cycle_src_loss
        
        #Identity loss
        idt_trg_loss = l1(idt_trg, trg)
        idt_src_loss = l1(idt_src, src)
        total_idt_loss = idt_trg_loss + idt_src_loss
        
        gen_loss = total_adv_loss + lambda_cycle * total_cycle_loss + lambda_cycle * idt_coef * total_idt_loss
            
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        loss_gen_ep.append(gen_loss.item())
        
        
        # Train discriminators
        fake_history_trg = buffer_trg(fake_trg.detach().clone())
        fake_history_trg.to(device)
        trg_real_pred = dis_trg(trg)
        trg_fake_pred = dis_trg(fake_history_trg)
        real_score_ep.append(trg_real_pred.mean().item())
        fake_score_ep.append(trg_fake_pred.mean().item())
        dis_trg_real_loss = mse(trg_real_pred, torch.ones_like(trg_real_pred))
        dis_trg_fake_loss = mse(trg_fake_pred, torch.zeros_like(trg_fake_pred))
        dis_trg_loss = dis_trg_real_loss + dis_trg_fake_loss
        
        fake_history_src = buffer_src(fake_src.detach().clone())
        fake_history_src.to(device)
        src_real_pred = dis_src(src)
        src_fake_pred = dis_src(fake_history_src)
        dis_src_real_loss = mse(src_real_pred, torch.ones_like(src_real_pred))
        dis_src_fake_loss = mse(src_fake_pred, torch.zeros_like(src_fake_pred))
        dis_src_loss = dis_src_real_loss + dis_src_fake_loss
        
        dis_loss = (dis_trg_loss + dis_src_loss) / 2
        
        opt_dis.zero_grad()
        dis_loss.backward()
        opt_dis.step()
        loss_dis_ep.append(dis_loss.item())
        
    scheduler_dis.step()
    scheduler_gen.step()
       
    gen_src.eval()
    gen_trg.eval()
    with torch.no_grad():
        val_src = val_src.to(device)
        val_trg = val_trg.to(device)
        val_trg_img = gen_trg(val_src).cpu().detach()
        val_src_img = gen_src(val_trg).cpu().detach()
    
    return np.mean(loss_dis_ep), np.mean(loss_gen_ep), np.mean(real_score_ep), \
           np.mean(fake_score_ep), val_trg_img, val_src_img