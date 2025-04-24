# pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
# model
from simba import Mmamba
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
import dadam as optim
import cosine_lr_scheduler as lr_scheduler

# others
from glob import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
import copy
import math
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    # continue or not
    parser.add_argument("-c", "--is_continue", action="store_true")
    
    parser.add_argument('--device', type=str,
                        help='gpu device.', default='cuda')
    parser.add_argument('--project_name', type=str,
                        help='project_name.', default='new_project')
    
    # about training
    parser.add_argument('--batch', type=int,
                        help='batch size', default=4)
    parser.add_argument('--accumulation_step', type=int,
                        help='accumulation_step', default=4)
    
    # about model
    parser.add_argument('--layer_num', type=int,
                        help='layers of model', default=24)
    parser.add_argument('--d_state', type=int,
                        help='state size of mamba', default=512)
    
    parser.add_argument('--ckpt', type=int,
                        help='ckpt epoch.', default=None)
    args = parser.parse_args()
    return args

opt = parse_opt()
print(opt)

def token_to_seq(tokens):
    '''
    INPUT:
    tokens: a encodec compressed token with 4 residual layers
    1 5 9... (course layer)
    2 6 10...
    3 7 11...
    4 8 12... (fine layer)
    
    OUTPUT:
    a: a flatten seq
    1 2 3 4 5 6 7 8 9 10 11 12...
    '''
    K, L = tokens.shape
    a = np.zeros((K*L))
    for i in range(K*L):
        a[i] = tokens[i%4, i//4]
    return a

def seq_to_token(seq):
    '''
    INPUT:
    a: a flatten seq
    1 2 3 4 5 6 7 8 9 10 11 12...
    
    OUTPUT:
    tokens: a encodec compressed token with 4 residual layers
    1 5 9... (course layer)
    2 6 10...
    3 7 11...
    4 8 12... (fine layer)
    '''
    L = seq.shape[0]
    print(L)
    a = np.zeros((4, L//4))
    idx = 0
    for i in range(L//4):
        for j in range(4):
            a[j][i] = seq[idx]
            idx+=1
    return a

def to_delay(para):
    '''
    INPUT:
    tokens: a k layers RVQ token
    1 5 9... (course layer)
    2 6 10...
    3 7 11...
    4 8 12... (fine layer)
    
    OUTPUT:
    delay: the delay pattern
    0 1 5 9... (course layer)
    0 0 2 6 10...
    0 0 0 3 7 11...
    0 0 0 0 4 8 12... (fine layer)
    '''
    K, L = para.shape
    delay = np.zeros((4, 500)) + 2048
    for i in range(K):
        # delay[i, i:] = para[i, :500-i]
        delay[i, i+1:] = para[i, :499-i]
    return delay

def to_parallel(delay):
    '''
    INPUT:
    delay: the delay pattern
    1 5 9... (course layer)
    0 2 6 10...
    0 0 3 7 11...
    0 0 0 4 8 12... (fine layer)
    
    
    OUTPUT:
    para: a k layers RVQ token w/ parallel form
    1 5 9... (course layer)
    2 6 10...
    3 7 11...
    4 8 12... (fine layer)
    '''
    K, L = delay.shape
    para = np.zeros((K, L)) + 2048
    for i in range(K):
        # para[i, :500-i] = delay[i, i:]
        para[i, :499-i] = delay[i, i+1:]
    return para

def cal_torch_model_params(model):
    '''
    :param model:
    :return:
    '''
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total_params/1000000, 'total_trainable_params': total_trainable_params/1000000}

class Dataset(object):
    '''
    Jamendo dataset
    '''
    def __init__(self, datalist) -> None:
        self.data_path = datalist
    
    def __getitem__(self, idx):
        # shape = [1, 128, slice_length]
        path = self.data_path[idx]
        data = np.load(path, allow_pickle=True)
        # return seq[:, 0:1499], seq[:, 1:1500]
        return torch.LongTensor(data[:, :-1]), torch.LongTensor(data[:, 1:])
    
    def __len__(self):
        return len(self.data_path)

class MB_Dataset(object):
    '''
    MusicBench dataset
    '''
    def __init__(self, metadata, root_path, L = 500) -> None:
        self.meta = metadata
        self.root = root_path
        self.length = L
        self.number = len(self.meta)
    
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.meta[idx]['location'][:-4]+'.npy')
        description = self.meta[idx]['main_caption']
        data = np.load(path, allow_pickle=True)
        K, L = data.shape
        data = torch.LongTensor(np.pad(data,((0, 0), (0, self.length-L)),'constant',constant_values=(0,0)))
        data = to_delay(data)
        # return seq[:, 0:499], seq[:, 1:500]
        return torch.LongTensor(data[:, :-1]), torch.LongTensor(data[:, 1:]), description
    
    def __len__(self):
        return len(self.meta)

def temperature_sampling(logits, temperature, topk):
        # probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        logits = torch.Tensor(logits)
        probs = nn.Softmax(dim=0)(logits / temperature)
        probs = np.array(probs)
        if topk == 1:
            prediction = np.argmax(probs)
        else:
            sorted_index = np.argsort(probs)[::-1]
            candi_index = sorted_index[:topk]
            candi_probs = [probs[i] for i in candi_index]
            # normalize probs
            candi_probs /= sum(candi_probs)
            # choose by predicted probs
            prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return prediction

def train():
    print('setup...', end='\r')
    
    ########################################################################
    # training
    EPOCH = 200
    start_epoch = 1
    BATCH = opt.batch
    project_name = opt.project_name
    max_grad_norm = 1
    device = opt.device
    accumulation_step = opt.accumulation_step
    dataset_type = 'MusicBench'
    ################################################
    # model config setup
    model_config = {
        'layers':opt.layer_num,
        'vocab_size':2048+1,
        'd_model':1024,
        'drop_p':0.2,
        'd_state':opt.d_state,
    }
    ################################################
    # optimizer config setup
    optim_config = {
        'optim_lr': 1e-4,
        'weight_decay':0.05,
        'betas': (0.9, 0.999),
    }
    ########################################################################
    
    # ckpts folder path
    os.makedirs('./ckpts', exist_ok=True)
    ckpt_folder = './ckpts/{}'.format(project_name)
    os.makedirs(ckpt_folder, exist_ok=True)
    
    # dataset
    # path = glob('/mnt/gestalt/home/lonian/datasets/mtg_crop_sep_token/*.npy')
    metadata_path = '/mnt/gestalt/home/lonian/datasets/MusicBench/musicbench_train_simba.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    train_data = MB_Dataset(metadata, root_path = '/mnt/gestalt/home/lonian/datasets/MusicBench/data_token')
    train_loader = DataLoader(dataset=train_data, batch_size = BATCH, shuffle=True, num_workers=4, pin_memory=True)
    
    # model and optimizer and scheduler
    print('Loading model...', end='\r')
    music_model = Mmamba(
        layers = model_config['layers'], 
        vocab_size = model_config['vocab_size'], 
        d_model = model_config['d_model'], 
        drop_p = model_config['drop_p'], 
        d_state = model_config['d_state'])
    
    music_model = music_model.to(device)
    
    
    config = {}
    config['training'] = {
        'name': project_name,
        'dataset': dataset_type, 
        'epoch': EPOCH,
        'data_number': len(metadata),
        'batch': BATCH,
        'accumulation_step': accumulation_step,
        'model_size': cal_torch_model_params(music_model)
    }
    config['model'] = model_config
    
    ##########################################################################################
    # # MUSICGEN optimizer and scheduler settings
    # optimizer = optim.DAdaptAdam(music_model.parameters(), lr=1)
    # total_updates = len(train_loader) * EPOCH
    # scheduler = lr_scheduler.CosineLRScheduler(optimizer, warmup_steps=len(train_loader)*2, total_steps=total_updates)
    # config['optimizer'] = {
    #     'lr': 1,
    # }
    # config['scheduler'] = {
    #     'warmup_steps': len(train_loader), 
    #     'total_steps': total_updates
    # }
    ##########################################################################################
    # original optimizer and scheduler settings
    optimizer = AdamW(  music_model.parameters(), 
                        lr = optim_config['optim_lr'], 
                        weight_decay = optim_config['weight_decay'], 
                        betas = optim_config['betas']    )
    # torch_lr_scheduler = CosineAnnealingLR(  optimizer, 
    #                                 T_max = EPOCH-10 )
    # scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler,
    #                                         warmup_start_value=0,
    #                                         warmup_end_value=optim_config['optim_lr'],
    #                                         warmup_duration=10)

    warm_up_iter = 10
    T_max = EPOCH	# 周期
    lr_max = 1e-1	# 最大值
    lr_min = 5e-4	# 最小值

    # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
    lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
    (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    config['optimizer'] = optim_config
    config['scheduler'] = {
        'warmup_duration': warm_up_iter,
        'T_max': EPOCH
    }
    ##########################################################################################
    
    with open(os.path.join(ckpt_folder, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    losses_list = []
    min_loss = 100
    for epoch in range(start_epoch, EPOCH+1):
        music_model.train()
        single_epoch = []
        iter_id = 1
        optimizer.zero_grad()
        losses = 0
        for x, y, text in tqdm(train_loader, ncols=120):
        # for x, y, text in train_loader:
        #     print('Iter: {:04d} / {}  | Loss = {:.4f}'.format(iter_id, len(train_loader), np.array(single_epoch).mean()*accumulation_step), end='\r')
            x = x.to(device)
            y = y.to(device)
            torch.cuda.set_device(x.device.index)
            out = music_model(x)
            output_logit = out
            # output_logit = out
            losses = 0
            # print(output_logit.shape) # [B, 4, 1499, 2048]
            # output_logit = output_logit.permute(0, 3, 1, 2)
            # print(output_logit.shape, y.shape) # [B, 2048, 4, 1499]
            for k in range(4):
                # print(y.shape)
                logits_k = output_logit[:, k, :, :].contiguous().view(-1, output_logit.size(-1))
                targets_k = y[:, k, :].contiguous().view(-1)
                loss = nn.CrossEntropyLoss(ignore_index=2048)(logits_k, targets_k)
                losses += loss
            
            losses = losses / (4*accumulation_step)
            # print(losses)
            losses.backward()
            
            if iter_id % accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(music_model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            single_epoch.append(losses.to('cpu').mean().item())
            iter_id += 1
        
        scheduler.step()
        
        single_epoch = np.array(single_epoch)
        losses_list.append(single_epoch.mean()*accumulation_step)
        print('>>> Epoch: {}, Loss: {:.5f}'.format(epoch, losses_list[-1]))
        
        
        if epoch % 2 == 0:
            torch.save({'epoch': epoch,
                        'model': music_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': losses_list[-1],
                        }, os.path.join(ckpt_folder, 'epoch_%03d.pkl'%epoch))
            # losses = np.array(losses)
        
        if losses_list[-1] < min_loss:
            torch.save({'epoch': epoch,
                        'model': music_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': losses_list[-1],
                        }, os.path.join(ckpt_folder, 'best.pkl'))
        
        np.save(os.path.join(ckpt_folder, 'training_loss'), np.array(losses_list))

def cont_train():
    print('CONTINUE setup...', end='\r')
    
    ########################################################################
    cont_epoch = opt.ckpt
    assert cont_epoch % 2 == 0
    # training
    EPOCH = 200
    start_epoch = 1
    project_name = opt.project_name
    device = opt.device
    
    max_grad_norm = 1
    dataset_type = 'MusicBench'
    ################################################
    with open('/mnt/gestalt/home/lonian/mamba/model/ckpts/{}/config.json'.format(project_name)) as f:
        config = json.load(f)
    # training config setup
    train_config = config['training']
    BATCH = train_config['batch']
    accumulation_step = train_config['accumulation_step']
    ################################################
    # model config setup
    model_config = config['model']
    ################################################
    # optimizer config setup
    optim_config = config['optimizer']
    ########################################################################
    
    # ckpts folder path
    ckpt_folder = '/mnt/gestalt/home/lonian/mamba/model/ckpts/{}'.format(project_name)
    os.makedirs(ckpt_folder, exist_ok=True)
    
    # dataset
    metadata_path = '/mnt/gestalt/home/lonian/datasets/MusicBench/musicbench_train_simba.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    train_data = MB_Dataset(metadata, root_path = '/mnt/gestalt/home/lonian/datasets/MusicBench/data_token')
    train_loader = DataLoader(dataset=train_data, batch_size = BATCH, shuffle=True, num_workers=4, pin_memory=True)
    
    # model and optimizer and scheduler
    print('Loading model...', end='\r')
    music_model = Mmamba(
        layers = model_config['layers'], 
        vocab_size = model_config['vocab_size'], 
        d_model = model_config['d_model'], 
        drop_p = model_config['drop_p'], 
        d_state = model_config['d_state'])
    
    music_model = music_model.to(device)
    
    
    
    optimizer = AdamW(  music_model.parameters(), 
                        lr = optim_config['optim_lr'], 
                        weight_decay = optim_config['weight_decay'], 
                        betas = optim_config['betas']    )

    warm_up_iter = 10
    T_max = EPOCH	# 周期
    lr_max = 1e-1	# 最大值
    lr_min = 5e-4	# 最小值

    # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
    lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
    (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    ##########################################################################################
    
    checkpoints_path = '/mnt/gestalt/home/lonian/mamba/model/ckpts/{}/epoch_{:03d}.pkl'.format(project_name, cont_epoch)
    if os.path.isfile(checkpoints_path):
        checkpoint = torch.load(checkpoints_path, map_location=device)
    else:
        os._exit()
    start_epoch = checkpoint['epoch'] + 1
    music_model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    
    losses_list = list(np.load('/mnt/gestalt/home/lonian/mamba/model/ckpts/{}/training_loss.npy'.format(project_name)))
    losses_list = losses_list[:cont_epoch]
    assert len(losses_list) == cont_epoch
    # print(len(losses_list), min(losses_list))
    min_loss = min(losses_list)
    
    for epoch in range(start_epoch, EPOCH+1):
        music_model.train()
        single_epoch = []
        iter_id = 1
        optimizer.zero_grad()
        for x, y, text in tqdm(train_loader, ncols=120):
            x = x.to(device)
            y = y.to(device)
            torch.cuda.set_device(x.device.index)
            out = music_model(x)
            output_logit = out
            # output_logit = out
            losses = 0
            
            for k in range(4):
                # print(y.shape)
                logits_k = output_logit[:, k, :, :].contiguous().view(-1, output_logit.size(-1))
                targets_k = y[:, k, :].contiguous().view(-1)
                
                loss = nn.CrossEntropyLoss(ignore_index=2048)(logits_k, targets_k)
                losses += loss
            
            losses = losses / (4*accumulation_step)
            # print(losses)
            losses.backward()
            
            if iter_id % accumulation_step == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(music_model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            single_epoch.append(losses.to('cpu').mean().item())
            iter_id += 1
        
        scheduler.step()
        
        single_epoch = np.array(single_epoch)
        losses_list.append(single_epoch.mean()*accumulation_step)
        print('>>> Epoch: {}, Loss: {:.4f}'.format(epoch, losses_list[-1]))
        
        
        if epoch % 2 == 0:
            torch.save({'epoch': epoch,
                        'model': music_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': losses_list[-1],
                        }, os.path.join(ckpt_folder, 'epoch_%03d.pkl'%epoch))
            # losses = np.array(losses)
        
        if losses_list[-1] < min_loss:
            torch.save({'epoch': epoch,
                        'model': music_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss': losses_list[-1],
                        }, os.path.join(ckpt_folder, 'best.pkl'))
        
        np.save(os.path.join(ckpt_folder, 'training_loss'), np.array(losses_list))

def main():
    if opt.is_continue:
        cont_train()
    else:
        train() 


if __name__ == '__main__':
    main()