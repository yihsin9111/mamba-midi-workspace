from simba import Mmamba
import os
import argparse

'''
README

生成 unconditinoal 的 code
只給定前面的 [2048 2048 2048 2048] code 沒有其他引導

SPECIAL_ID = 0     :適用在 v24 之前
SPECIAL_ID = 2048  :適用在 v24 之後
需要調整temp, topk
'''

SPECIAL_ID = 2048

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,
                        help='gpu device.', default='cuda:2')
    parser.add_argument('--project_name', type=str,
                        help='project_name.', default='v19')
    parser.add_argument('--ckpt', type=int,
                        help='ckpt epoch.', default=50)
    parser.add_argument('--num', type=int,
                        help='how many samples you want to generate', default=10)
    parser.add_argument('--batch', type=int,
                        help='how many samples you want to generate', default=1)
    args = parser.parse_args()
    return args

def create_empty_prompt(num):
    '''
    INPUT:
    num: batch number
    
    OUTPUT:
    empty tensor w/ [batch, 4, 1] shape
    '''
    prompt = np.zeros((num, 4, 1)) + SPECIAL_ID
    return prompt

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
    delay = np.zeros((4, 500)) + SPECIAL_ID
    for i in range(K):
        # delay[i, i:] = para[i, :500-i]
        delay[i, i+1:] = para[i, :499-i]
    return delay

def to_parallel(delay):
    '''
    INPUT:
    delay: the delay pattern
    0 1 5 9... (course layer)
    0 0 2 6 10...
    0 0 0 3 7 11...
    0 0 0 0 4 8 12... (fine layer)
    
    
    OUTPUT:
    para: a k layers RVQ token w/ parallel form
    1 5 9... (course layer)
    2 6 10...
    3 7 11...
    4 8 12... (fine layer)
    '''
    K, L = delay.shape
    para = np.zeros((K, L)) + SPECIAL_ID
    for i in range(K):
        para[i, :L-i-1] = delay[i, i+1:]
    return para

opt = parse_opt()
print(opt)

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
model_path = '/mnt/gestalt/home/lonian/mamba/model/ckpts/{}/epoch_{:03d}.pkl'.format(opt.project_name, opt.ckpt)
save_path = '/mnt/gestalt/home/lonian/mamba/model/{}_results/{}_noprompt'.format(opt.project_name, opt.ckpt)
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


from tqdm import tqdm
import numpy as np
single_gen = opt.batch
gen_num = 1
for idx in tqdm(range(0, opt.num, 1)):
    prompt_seq = create_empty_prompt(single_gen)
    
    input_seq = torch.LongTensor(prompt_seq).to(device)
    torch.cuda.set_device(input_seq.device.index)
    B, K, L = prompt_seq.shape
    # print(prompt_seq.shape, 'generate !')
    while L < 800+5:
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
                        temperature=1.2,
                        topk=200)
                words.append([word])
            batch_new.append(words)
        prompt_seq = np.concatenate((prompt_seq, batch_new), axis=2)
        L+=1
    # print(prompt_seq.shape)
    # prompt_seq = to_parallel(prompt_seq)
    
    for b in range(single_gen):
        p_seq = to_parallel(prompt_seq[b])
        np.save(os.path.join(save_path, '{}.npy'.format(gen_num)), p_seq[:, 300:800])
        gen_num += 1