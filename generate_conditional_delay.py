from simba import Mmamba, Text_Mmamba
import os
import argparse
from transformers import AutoTokenizer, T5Tokenizer
from transformers import T5EncoderModel

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
                        help='project_name.', default='text_v4')
    parser.add_argument('--ckpt', type=int,
                        help='ckpt epoch.', default=50)
    
    parser.add_argument('--num', type=int,
                        help='how many samples you want to generate', default=10)
    parser.add_argument('--batch', type=int,
                        help='how many samples you want to generate', default=10)
    
    parser.add_argument('--temp', type=float,
                        help='temparature', default=1.2)
    parser.add_argument('--topk', type=int,
                        help='topk', default=250)
    parser.add_argument('--g_scale', type=float,
                        help='condition guidance scale', default=3)
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

music_model = Text_Mmamba(
        layers = model_config['layers'],
        vocab_size = model_config['vocab_size'],
        d_model = model_config['d_model'],
        drop_p = model_config['drop_p'], 
        d_state = model_config['d_state'], 
        num_heads = model_config['num_heads'], 
        inner = model_config['inner'], 
        self_atten_layers = model_config['self_atten_layers'])

music_model = music_model.to(device)

import torch
model_path = '/mnt/gestalt/home/lonian/mamba/model/ckpts/{}/epoch_{:03d}.pkl'.format(opt.project_name, opt.ckpt)
save_path_A = '/mnt/gestalt/home/lonian/mamba/model/{}_results/{}_A'.format(opt.project_name, opt.ckpt)
save_path_B = '/mnt/gestalt/home/lonian/mamba/model/{}_results/{}_B'.format(opt.project_name, opt.ckpt)
os.makedirs(save_path_A, exist_ok=True)
os.makedirs(save_path_B, exist_ok=True)

checkpoint = torch.load(model_path, map_location='cpu')
print(checkpoint['loss'])
music_model.load_state_dict(checkpoint['model'])
music_model.eval()

# text encoder
text_encoder_name = 'google/flan-t5-base'
tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)
text_encoder = T5EncoderModel.from_pretrained(text_encoder_name).train(mode=False)
text_encoder = text_encoder.to(device)

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

from glob import glob
with open('/mnt/gestalt/home/lonian/datasets/MusicBench/MusicBench_test_A_simba.json') as f:
    meta_A = json.load(f)

with open('/mnt/gestalt/home/lonian/datasets/MusicBench/MusicBench_test_B_simba.json') as f:
    meta_B = json.load(f)

from tqdm import tqdm
import numpy as np
single_gen = opt.batch
gen_num = 1
A_config = {
    'temp': opt.temp,
    'topk': opt.topk,
    'guidance_scale': opt.g_scale,
    'description': {}
}
B_config = {
    'temp': opt.temp,
    'topk': opt.topk,
    'guidance_scale': opt.g_scale,
    'description': {}
}
for idx in tqdm(range(0, opt.num, 1)):
    # ABAB0000
    description = []
    for n in range(single_gen):
        try:
            description.append(meta_A[idx*single_gen + n]['main_caption'])
            description.append(meta_B[idx*single_gen + n]['main_caption'])
            A_config['description'][idx*single_gen + n] = meta_A[idx*single_gen + n]['main_caption']
            B_config['description'][idx*single_gen + n] = meta_B[idx*single_gen + n]['main_caption']
            # A_config['description'].append(meta_A[idx*single_gen + n]['main_caption'])
            # B_config['description'].append(meta_B[idx*single_gen + n]['main_caption'])
        except:
            break
    for n in range(single_gen):
        try:
            description.append('')
            description.append('')
        except:
            break
    
    prompt_seq = create_empty_prompt(len(description)//2)
    input_seq = torch.LongTensor(prompt_seq).to(device)
    
    # process text
    batch = tokenizer(
        description, padding=True, return_tensors="pt"
    )
    input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

    with torch.set_grad_enabled(False):
        text_embedding = text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
    
    text_embedding_mask = (attention_mask == 1).to(device)
    
    torch.cuda.set_device(input_seq.device.index)
    B, K, L = prompt_seq.shape
    # print(prompt_seq.shape, 'generate !')
    while L < 500+10:
        # print(L, prompt_seq.shape, end='\r')
        
        with torch.no_grad():
            cond_output_logits = music_model(torch.LongTensor(prompt_seq).to(device), text_embedding[:len(description)//2], text_embedding_mask[:len(description)//2])
            uncond_output_logits = music_model(torch.LongTensor(prompt_seq).to(device), text_embedding[len(description)//2:], text_embedding_mask[len(description)//2:])
        # output_logits = model(torch.LongTensor(np.array([prompt_seq])).to(device))
        # print(output_logits.shape) # [B, 4, L+1, 2048]
        output_logits = uncond_output_logits + (cond_output_logits - uncond_output_logits) * opt.g_scale
        # logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg_coef
        _logit = output_logits[:, :, -1, :].to('cpu').detach().numpy()
        batch_new = []
        for b in range(B):
            words = []
            for i in range(4):
                word = temperature_sampling(
                        logits=_logit[b, i],
                        temperature=opt.temp,
                        topk=opt.topk)
                words.append([word])
            batch_new.append(words)
        prompt_seq = np.concatenate((prompt_seq, batch_new), axis=2)
        L+=1
    # print(prompt_seq.shape)
    # prompt_seq = to_parallel(prompt_seq)
    
    for b in range(len(prompt_seq)):
        p_seq = to_parallel(prompt_seq[b])
        if b%2 == 0:
            np.save(os.path.join(save_path_A, '{}.npy'.format(b//2+idx*single_gen)), p_seq[:, :500])
        else:
            np.save(os.path.join(save_path_B, '{}.npy'.format(b//2+idx*single_gen)), p_seq[:, :500])

with open(os.path.join(save_path_A, 'descriptions.json'), 'w', encoding='utf-8') as f:
    json.dump(A_config, f, ensure_ascii=False, indent=4)
with open(os.path.join(save_path_B, 'descriptions.json'), 'w', encoding='utf-8') as f:
    json.dump(B_config, f, ensure_ascii=False, indent=4)