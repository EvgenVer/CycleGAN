import numpy as np
import torch
from tqdm.auto import tqdm

def train_epoch(gen_trg, gen_src, dis_trg, dis_src, 
                loader, opt_gen, opt_dis, l1, mse, 
                dis_scal, gen_scal, lambda_cycle, device,
                scheduler_dis, scheduler_gen, buffer_trg, buffer_src):
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
        
        # Train discriminators
        with torch.cuda.amp.autocast():
            fake_trg = gen_trg(src)
            if np.random.random() > 0.5:
                trg_idx = np.random.randint(len(buffer_trg))
                fake_history_trg = buffer_trg.pop(trg_idx)
                buffer_trg.append(fake_trg.detach().clone())
            else:
                fake_history_trg = fake_trg.detach().clone()
            fake_history_trg.to(device)
            trg_real_pred = dis_trg(trg)
            trg_fake_pred = dis_trg(fake_history_trg)
            real_score_ep.append(trg_real_pred.mean().item())
            fake_score_ep.append(trg_fake_pred.mean().item())
            dis_trg_real_loss = mse(trg_real_pred, torch.ones_like(trg_real_pred))
            dis_trg_fake_loss = mse(trg_fake_pred, torch.zeros_like(trg_fake_pred))
            dis_trg_loss = dis_trg_real_loss + dis_trg_fake_loss
            
            fake_src = gen_src(trg)
            if np.random.random() > 0.5:
                src_idx = np.random.randint(len(buffer_src))
                fake_history_src = buffer_src.pop(src_idx)
                buffer_src.append(fake_src.detach().clone())
            else:
                fake_history_src = fake_src.detach().clone()
            fake_history_src.to(device)
            src_real_pred = dis_src(src)
            src_fake_pred = dis_src(fake_history_src)
            dis_src_real_loss = mse(src_real_pred, torch.ones_like(src_real_pred))
            dis_src_fake_loss = mse(src_fake_pred, torch.zeros_like(src_fake_pred))
            dis_src_loss = dis_src_real_loss + dis_src_fake_loss
            
            dis_loss = (dis_trg_loss + dis_src_loss) / 2
        
        opt_dis.zero_grad()
        dis_scal.scale(dis_loss).backward()
        dis_scal.step(opt_dis)
        dis_scal.update()
        loss_dis_ep.append(dis_loss.item())
        
        # Train generators
        with torch.cuda.amp.autocast():
            trg_fake_pred = dis_trg(fake_trg)
            src_fake_pred = dis_src(fake_src)
            gen_trg_loss = mse(trg_fake_pred, torch.ones_like(trg_fake_pred))
            gen_src_loss = mse(src_fake_pred, torch.ones_like(src_fake_pred))
            gen_adv_loss = (gen_trg_loss + gen_src_loss) / 2
            
            cycle_trg = gen_trg(fake_src)
            cycle_src = gen_src(fake_trg)
            cycle_trg_loss = l1(cycle_trg, trg)
            cycle_src_loss = l1(cycle_src, src)
            cycle_loss = (cycle_trg_loss + cycle_src_loss) / 2
            
            gen_loss = gen_adv_loss + lambda_cycle * cycle_loss
            
        opt_gen.zero_grad()
        gen_scal.scale(gen_loss).backward()
        gen_scal.step(opt_gen)
        gen_scal.update()
        loss_gen_ep.append(gen_loss.item())
        
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
           np.mean(fake_score_ep), val_trg_img, val_src_img, buffer_trg, buffer_src