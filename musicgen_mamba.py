from transformers import MambaConfig, MambaModel
from glob import glob
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
import copy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

import dadam as optim
import cosine_lr_scheduler as scheduler

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
    def __init__(self):
        super(Musicmamba, self).__init__()
        # parameters setup
        self.card = 2048
        self.dim = 1024
        self.mamba_layer = 48
        self.token_layer = 4
    
        configuration = MambaConfig(
            vocab_size = self.card,
            hidden_size = self.dim
        )
        
        self.emb = nn.ModuleList([ScaledEmbedding(self.card, self.dim) for _ in range(self.token_layer)])
        base_mamba_layer = MambaModel(configuration).layers[0]
        self.layers = nn.ModuleList([copy.deepcopy(base_mamba_layer) for _ in range(self.mamba_layer)])
        self.norm_f = MambaModel(configuration).norm_f
        
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
    EPOCH = 500
    start_epoch = 1
    BATCH = 4
    project_name = 'v9'
    
    optim_lr = 1
    max_grad_norm = 1
    warmup_steps = 4000
    ckpt_folder = '/mnt/gestalt/home/lonian/mamba/model/ckpts/{}'.format(project_name)
    
    device = 'cuda:2'
    
    os.makedirs(ckpt_folder, exist_ok=True)
    
    # dataset
    path = glob('/mnt/gestalt/home/lonian/datasets/mamba_test_token/*.npy')
    train_data = Dataset(path)
    train_loader = DataLoader(dataset=train_data, batch_size = BATCH, shuffle=True, num_workers=8, pin_memory=True)
    
    # model and optimizer
    print('Loading model...', end='\r')
    model = Musicmamba()
    
    config = {}
    model_config = model.config()
    config['model'] = model_config
    config['model']['structure'] = str(model.modules)
    config['model']['parameters'] = cal_torch_model_params(model)
    config['training'] = {'epoch': EPOCH}
    model = model.to(device)
    
    # optimizer = torch.optim.Adam(model.parameters())
    # v5
    config['optimizer'] = {'name': 'DAdaptAdam'}
    config['optimizer']['lr'] = optim_lr
    config['scheduler'] = {'name': 'CosineLRScheduler'}
    config['scheduler']['warmup_steps'] = warmup_steps
    optimizer = optim.DAdaptAdam(model.parameters(), lr=optim_lr)
    total_updates = len(train_loader) * EPOCH
    config['scheduler']['total_updates'] = total_updates
    lr_scheduler = scheduler.CosineLRScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_updates)
    
    
    # config['optimizer'] = {'name': 'AdamW'}
    # config['scheduler'] = {'name': 'CosineAnnealingLR'}
    # optimizer = torch.optim.AdamW(model.parameters(), lr=optim_lr)
    # # total_updates = (len(train_loader)//BATCH) * EPOCH
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6)
    
    with open(os.path.join(ckpt_folder, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    # return
    losses_list = []
    min_loss = 100
    for epoch in range(start_epoch, EPOCH+1):
        model.train()
        single_epoch = []
        
        for x, y in tqdm(train_loader, ncols=120):
            output_logit = model(x.to(device))
            y = y.to(device)
            losses = 0
            # print(output_logit.shape) # [B, 4, 1499, 2048]
            output_logit = output_logit.permute(0, 3, 1, 2)
            # print(output_logit.shape, y.shape) # [B, 2048, 4, 1499]
            for k in range(4):
                # print(y.shape)
                loss = nn.CrossEntropyLoss()(output_logit[:, :, k, :], y[:, k, :])
                losses += loss
            losses = losses / 4
            # print(losses)
            # print('\n======================================={}=========================================\n'.format(losses))
            optimizer.zero_grad()
            # 梯度裁剪
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            
            single_epoch.append(losses.to('cpu').mean().item())
            # break
        
        single_epoch = np.array(single_epoch)
        losses_list.append(single_epoch.mean())
        print('>>> Epoch: {}, Loss: {:.5f}'.format(epoch, losses_list[-1]))
        
        
        if epoch % 2 == 0:
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict(),
                        'loss': losses_list[-1],
                        }, os.path.join(ckpt_folder, 'epoch_%03d.pkl'%epoch))
            # losses = np.array(losses)
        
        if losses_list[-1] < min_loss:
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict(),
                        'loss': losses_list[-1],
                        }, os.path.join(ckpt_folder, 'best.pkl'))
        
        np.save(os.path.join(ckpt_folder, 'training_loss'), np.array(losses_list))

def test():
    # print('test')
    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.makedirs('./results', exist_ok=True)
    model_path = '/mnt/gestalt/home/lonian/mamba/model/ckpts/v9/epoch_230.pkl'
    # model_path = '/mnt/gestalt/home/lonian/mamba/model/ckpts/v9/best.pkl'
    idx = len(glob('/mnt/gestalt/home/lonian/mamba/model/results/*.npy'))+1
    device = 'cuda:1'
    temperature = 1.2
    topk = 200
    prompt_length = 100
    
    
    with torch.no_grad():
        # load prompt
        prompt_id = [2]
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
        model = Musicmamba().to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        
        
        
        # prompt_seq = token_to_seq(prompt)
        print(prompt_seq.shape)
        B, K, L = prompt_seq.shape
        # z = np.zeros((4, 1500-prompt_length))
        # prompt_seq = np.concatenate((z, prompt_seq), axis=1)
        
        
        # prompt_seq = torch.LongTensor(prompt_seq).to(device)
        while L < 1500:
            # input_seq = prompt_seq[:, -1499:]
            print(L, prompt_seq.shape, end='\r')
            # print(input_seq.shape)
            # if L == prompt_length:
            #     output_logits = model(torch.LongTensor(prompt_seq).to(device))
            #     print(prompt_seq.shape)
            # else:
            #     output_logits = model(torch.LongTensor(prompt_seq[:, :, -1:]).to(device))
            output_logits = model(torch.LongTensor(prompt_seq).to(device))
            # output_logits = model(torch.LongTensor(np.array([prompt_seq])).to(device))
            # print(output_logits.shape) # [B, 4, L+1, 2048]
            _logit = output_logits[:, :, -1, :].to('cpu').detach().numpy()
            batch_new = []
            for b in range(B):
                words = []
                for i in range(4):
                    word = temperature_sampling(
                            logits=_logit[b, i],
                            temperature=temperature,
                            topk=topk)
                    words.append([word])
                batch_new.append(words)
            prompt_seq = np.concatenate((prompt_seq, batch_new), axis=2)
            # L = prompt_seq.shape[1]
            L+=1
    # tokens = seq_to_token(prompt_seq)
    for b in range(B):
        np.save('/mnt/gestalt/home/lonian/mamba/model/results/{}.npy'.format(idx), prompt_seq[b])
        idx+=1
    print('\n', prompt_seq.shape)
    pass


def main():
    test()


if __name__ == '__main__':
    main()