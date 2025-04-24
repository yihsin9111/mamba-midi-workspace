# pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
# model
# from transformers import MambaModel
# from transformers import MambaConfig as MC1
from mamba_ssm import Mamba2 # type: ignore
from models import MusicMambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig # type: ignore
'''
    class MambaConfig:
        d_model: int = 2560
        d_intermediate: int = 0
        n_layer: int = 64
        vocab_size: int = 50277
        ssm_cfg: dict = field(default_factory=dict)
        attn_layer_idx: list = field(default_factory=list)
        attn_cfg: dict = field(default_factory=dict)
        rms_norm: bool = True
        residual_in_fp32: bool = True
        fused_add_norm: bool = True
        pad_vocab_size_multiple: int = 8
        tie_embeddings: bool = True
    '''
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).
    """
    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group

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

def token_delay(tokens):
    pass

def delay_to_token(delay_tokens):
    pass

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
    def __init__(self, datalist) -> None:
        self.data_path = datalist
    
    def __getitem__(self, idx):
        # shape = [1, 128, slice_length]
        path = self.data_path[idx]
        data = np.load(path, allow_pickle=True)
        # return seq[:, 0:1499], seq[:, 1:1500]
        return 	torch.LongTensor(data[:, :-1]), torch.LongTensor(data[:, 1:])
    
    def __len__(self):
        return len(self.data_path)

class Musicmamba(nn.Module):
    def __init__(self, config):
        super(Musicmamba, self).__init__()
        # parameters setup
        self.card = config.vocab_size
        self.dim = 1024
        self.mamba_layer = 48
        self.token_layer = 4
    
        configuration = MambaConfig(
            vocab_size = self.card,
            hidden_size = self.dim
        )
        
        self.emb = nn.ModuleList([ScaledEmbedding(self.card, self.dim) for _ in range(self.token_layer)])
        base_mamba_layer = Mamba2(configuration).layers[0]
        self.layers = nn.ModuleList([copy.deepcopy(base_mamba_layer) for _ in range(self.mamba_layer)])
        
        self.linears = nn.ModuleList([nn.Linear(self.dim, self.card) for _ in range(self.token_layer)])
    
    def config(self):
        config = {   
                    'card': self.card,
                    'mamba_layer': self.mamba_layer,
                    'token_layer': self.token_layer,
                    'dim': self.dim,
                }
        return config

    def forward(self, sequence):
        B, K, S = sequence.shape
        sequence = sum([self.emb[k](sequence[:, k]) for k in range(K)])
        for module in self.layers:
            sequence = module(sequence)
        out = self.norm_f(sequence)
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1)  # [B, K, S, card]
        return logits

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
    BATCH = 4
    project_name = 'v17'
    max_grad_norm = 1
    device = 'cuda:1'
    accumulation_step = 4
    ################################################
    # model config setup
    model_config = MambaConfig()
    model_config.n_layer = 36
    model_config.attn_layer_idx = [8, 17, 26, 35]
    # model_config.attn_layer_idx = [7, 15, 23, 31, 39, 47]
    # model_config.attn_layer_idx = [5, 11, 17, 23, 29, 35, 41, 47]
    model_config.attn_cfg = {'num_heads': 8}
    model_config.vocab_size = 2048
    model_config.d_model = 1024
    model_config.ssm_cfg = {"layer": "Mamba2", "d_state": 128}
    ################################################
    # optimizer config setup
    optim_config = {
        'optim_lr': 1e-6,
        'weight_decay':0.05,
        'betas': (0.9, 0.999),
    }
    ########################################################################
    
    # ckpts folder path
    ckpt_folder = '/mnt/gestalt/home/lonian/mamba/model/ckpts/{}'.format(project_name)
    os.makedirs(ckpt_folder, exist_ok=True)
    
    # dataset
    path = glob('/mnt/gestalt/home/lonian/datasets/maestro_token/*.npy')
    train_data = Dataset(path)
    train_loader = DataLoader(dataset=train_data, batch_size = BATCH, shuffle=True, num_workers=8, pin_memory=True)
    
    # model and optimizer and scheduler
    print('Loading model...', end='\r')
    music_model = MusicMambaLMHeadModel(model_config)
    music_model = music_model.to(device)
    # music_model.apply(weights_init)
    
    
    config = {}
    config['training'] = {
        'name': project_name,
        'epoch': EPOCH,
        'data_number': len(path),
        'batch': BATCH,
        'accumulation_step': accumulation_step,
        'model_size': cal_torch_model_params(music_model)
    }
    # config['model'] = str(model_config)
    config['model'] = { 'd_model': model_config.d_model,
                        'd_intermediate': model_config.d_intermediate, 
                        'n_layer': model_config.n_layer, 
                        'vocab_size': model_config.vocab_size, 
                        'ssm_cfg': model_config.ssm_cfg, 
                        'attn_layer_idx': model_config.attn_layer_idx, 
                        'attn_cfg': model_config.attn_cfg, 
                        'rms_norm': model_config.rms_norm, 
                        'residual_in_fp32': model_config.residual_in_fp32, 
                        'fused_add_norm': model_config.fused_add_norm, 
                        'pad_vocab_size_multiple': model_config.pad_vocab_size_multiple, 
                        'tie_embeddings': model_config.tie_embeddings   }
    
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
    scheduler = CosineAnnealingLR(  optimizer, 
                                    T_max = len(train_loader) * EPOCH )
    config['optimizer'] = optim_config
    config['scheduler'] = {
        'T_max': len(train_loader) * EPOCH
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
        for x, y in tqdm(train_loader, ncols=120):
        # for x, y in train_loader:
            # losses = 0
            # print('{} / {}\tLoss = {}'.format(iter_id, len(train_loader), losses.item()), end='\r')
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            torch.cuda.set_device(x.device.index)
            out = music_model(x)
            output_logit = out.logits
            # output_logit = out
            losses = 0
            # print(output_logit.shape) # [B, 4, 1499, 2048]
            output_logit = output_logit.permute(0, 3, 1, 2)
            # print(output_logit.shape, y.shape) # [B, 2048, 4, 1499]
            for k in range(4):
                # print(y.shape)
                loss = nn.CrossEntropyLoss()(output_logit[:, :, k, :], y[:, k, :])
                losses += loss
            # losses = losses / 4
            # # print('\n', losses)
            # # 梯度裁剪
            # losses.backward()
            # # nn.utils.clip_grad_value_(music_model.parameters(), max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(music_model.parameters(), max_grad_norm)
            # optimizer.step()
            # scheduler.step()
            
            # single_epoch.append(losses.to('cpu').mean().item())
            # print(losses)
            losses = losses / (4*accumulation_step)
            # print(losses)
            losses.backward()
            
            if iter_id % accumulation_step == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(music_model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
            
            single_epoch.append(losses.to('cpu').mean().item())
            iter_id += 1
        
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

def test():
    ################################################
    # sampling setup
    device = 'cuda:2'
    temperature = 1.2
    topk = 200
    topp = 0.95
    prompt_length = 100
    model_path = '/mnt/gestalt/home/lonian/mamba/model/ckpts/v10/epoch_180.pkl'
    ################################################
    # model config setup
    model_config = MambaConfig()
    model_config.n_layer = 48
    model_config.attn_layer_idx = [11, 23, 35, 47]
    model_config.attn_cfg = {'num_heads': 16}
    model_config.vocab_size = 2048
    model_config.d_model = 1024
    model_config.ssm_cfg = {"layer": "Mamba2", "d_state":512}
    ################################################
    os.makedirs('./results', exist_ok=True)
    # model_path = '/mnt/gestalt/home/lonian/mamba/model/ckpts/v9/best.pkl'
    idx = len(glob('/mnt/gestalt/home/lonian/mamba/model/results/*.npy'))+1
    
    with torch.no_grad():
        # load prompt
        prompt_id = [2]
        B = len(prompt_id)
        prompt = []
        for i in prompt_id:
            a = np.load('/mnt/gestalt/home/lonian/datasets/mamba_test_token/{}.npy'.format(i), allow_pickle=True)
            prompt.append(a)
        # print(prompt)
        prompt = np.array(prompt)
        prompt_seq = prompt[:, :, :prompt_length]
        
        # prompt = np.load('/mnt/gestalt/home/lonian/datasets/mamba_test_token/2.npy', allow_pickle=True)
        # prompt_seq = prompt[:, :prompt_length]
        
        # load model
        checkpoint = torch.load(model_path, map_location={'cuda:2': device})
        # checkpoint = torch.load(model_path)
        music_model = MusicMambaLMHeadModel(model_config)
        music_model = music_model.to(device)
        music_model.load_state_dict(checkpoint['model'])
        music_model.eval()
        
        input_seq = torch.LongTensor(prompt_seq).to(device)
        torch.cuda.set_device(input_seq.device.index)
        out = music_model.generate( input_ids=input_seq, 
                                    max_length=1500, 
                                    temperature=temperature, 
                                    top_k=topk, 
                                    top_p = topp    )
    
    for b in range(B):
        np.save('/mnt/gestalt/home/lonian/mamba/model/results/{}.npy'.format(idx), out[b])
        idx+=1
    print('\n', prompt_seq.shape)
    pass


def main():
    train()


if __name__ == '__main__':
    main()