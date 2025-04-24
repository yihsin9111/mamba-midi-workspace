from simba import Mmamba
import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,
                        help='gpu device.', default='cuda:2')
    parser.add_argument('--project_name', type=str,
                        help='project_name.', default='v19')
    parser.add_argument('--ckpt', type=int,
                        help='ckpt epoch.', default=50)
    args = parser.parse_args()
    return args

opt = parse_opt()
print(opt)

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
    1 5 9... (course layer)
    0 2 6 10...
    0 0 3 7 11...
    0 0 0 4 8 12... (fine layer)
    '''
    B, K, L = para.shape
    delay = np.zeros((B, K, L))
    for i in range(K):
        delay[:, i, i:] = para[:, i, :L-i]
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
    B, K, L = delay.shape
    para = np.zeros((B, K, L))
    for i in range(K):
        para[:, i, :L-i] = delay[:, i, i:]
    return para



device = opt.device
import json

with open('/mnt/gestalt/home/lonian/mamba/model/ckpts/{}/config.json'.format(opt.project_name)) as f:
    meta = json.load(f)
    
model_config = meta['model']

music_model = Mmamba(
        layers = model_config['layers'], 
        vocab_size = model_config['vocab_size'], 
        d_model = model_config['d_model'], 
        drop_p = model_config['drop_p'], 
        d_state = model_config['d_state'])

music_model = music_model.to(device)

import torch
model_path = '/mnt/gestalt/home/lonian/mamba/model/ckpts/{}/epoch_0{}.pkl'.format(opt.project_name, opt.ckpt)
save_path = '/mnt/gestalt/home/lonian/mamba/model/{}_results/{}_j'.format(opt.project_name, opt.ckpt)
os.makedirs(save_path, exist_ok=True)

checkpoint = torch.load(model_path, map_location='cpu')
print(checkpoint['loss'])
music_model.load_state_dict(checkpoint['model'])
music_model.eval()

from torch import nn

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

from glob import glob
import random
import json


idx = len(glob(os.path.join(save_path, '*_all.npy'))) + 1
# print(idx)
from tqdm import tqdm
for _ in tqdm(range(50)):
    import numpy as np
    # prompt_id = glob('/mnt/gestalt/home/lonian/datasets/maestro_token/*.npy')
    # prompt_id = glob('/mnt/gestalt/home/lonian/datasets/MusicBench/FMACaps_eval_set/tokendata/*.npy')
    prompt_id = glob('/mnt/gestalt/home/lonian/datasets/mtg_crop_sep_token/*.npy')
    prompt = []
    for i in range(3):
        a = np.load(random.choice(prompt_id), allow_pickle=True)
        a = a[:, :100]
        # print(a.shape)
        prompt.append(a)
    # print(prompt)
    prompt = np.array(prompt, dtype=object)
    # prompt_seq = prompt[:, :, :100]
    # convert to delay
    prompt_seq = to_delay(prompt)

    input_seq = torch.LongTensor(prompt_seq).to(device)
    torch.cuda.set_device(input_seq.device.index)

    B, K, L = prompt_seq.shape
    while L < 500+100:
        # print(L, prompt_seq.shape, end='\r')
        output_logits = music_model(torch.LongTensor(prompt_seq).to(device))
        # output_logits = model(torch.LongTensor(np.array([prompt_seq])).to(device))
        # print(output_logits.shape) # [B, 4, L+1, 2048]
        _logit = output_logits[:, :, -1, :].to('cpu').detach().numpy()
        batch_new = []
        for b in range(B):
            words = []
            for i in range(4):
                word = temperature_sampling(
                        logits=_logit[b, i],
                        temperature=1.0,
                        topk=250)
                words.append([word])
            batch_new.append(words)
        prompt_seq = np.concatenate((prompt_seq, batch_new), axis=2)
        L+=1

    prompt_seq = to_parallel(prompt_seq)
    
    for b in range(B):
        np.save(os.path.join(save_path, '{}_all.npy'.format(idx)), prompt_seq[b])
        np.save(os.path.join(save_path, '{}.npy'.format(idx)), prompt_seq[b, :, 100:])
        idx+=1
    if len(os.path.join(save_path, '*_all.npy'))>=150:
        break

