from simba import Mmamba
import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,
                        help='gpu device.', default='cuda:2')
    parser.add_argument('--ckpt', type=int,
                        help='ckpt epoch.', default=74)
    args = parser.parse_args()
    return args

opt = parse_opt()
print(opt)

device = opt.device
model_config = {
        'layers':40,
        'vocab_size':2048,
        'd_model':1024,
        'drop_p':0.2,
        'd_state':128,
        }

music_model = Mmamba(
        layers = model_config['layers'], 
        vocab_size = model_config['vocab_size'], 
        d_model = model_config['d_model'], 
        drop_p = model_config['drop_p'], 
        d_state = model_config['d_state'])

music_model = music_model.to(device)

import torch
model_path = '/mnt/gestalt/home/lonian/mamba/model/ckpts/v18/epoch_0{}.pkl'.format(opt.ckpt)
save_path = '/mnt/gestalt/home/lonian/mamba/model/v18_results/{}'.format(opt.ckpt)
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

idx = len(glob(os.path.join(save_path, '*.npy'))) + 1
# print(idx)
from tqdm import tqdm
for _ in tqdm(range(50)):
    import numpy as np
    # prompt_id = glob('/mnt/gestalt/home/lonian/datasets/maestro_token/*.npy')
    prompt_id = glob('/mnt/gestalt/home/lonian/datasets/mtg_crop_sep_token/*.npy')
    prompt = []
    for i in range(3):
        a = np.load(random.choice(prompt_id), allow_pickle=True)
        prompt.append(a)
    # print(prompt)
    prompt = np.array(prompt)
    prompt_seq = prompt[:, :, :100]

    input_seq = torch.LongTensor(prompt_seq).to(device)
    torch.cuda.set_device(input_seq.device.index)

    B, K, L = prompt_seq.shape
    while L <= 500:
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
                        topk=100)
                words.append([word])
            batch_new.append(words)
        prompt_seq = np.concatenate((prompt_seq, batch_new), axis=2)
        L+=1


    for b in range(B):
        np.save(os.path.join(save_path, '{}.npy'.format(idx)), prompt_seq[b])
        idx+=1

