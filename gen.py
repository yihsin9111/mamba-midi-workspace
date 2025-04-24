import os
import sys
# pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from pl_model import Text_Mmamba_pl
import argparse
# import lightning as L
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# others
from glob import glob
import numpy as np
# import os
import json
from tqdm import tqdm
import math
# import argparse
from transformers import T5EncoderModel, T5Tokenizer
# from text_simba import MB_Dataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# torch.multiprocessing.set_start_method('spawn')
from utils import *


def parse_opt():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--folder_name', type=str,
                        help='main folder name under exp folder', default='test')
    parser.add_argument('--mode', type=str,
                        help='sub folder name under folder_name folder', default='test')
    
    parser.add_argument('--config_pth', type=str, 
                        help='model config pth', required=True)
    parser.add_argument('--ckpt_pth', type=str, 
                        help='model ckpt pth', required=True)
    
    
    args = parser.parse_args()
    return args

opt = parse_opt()


def create_logger(logger_file_path, name=None):
    import time
    import logging
    
    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)
    if name is not None:
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    else:
        log_name = '{}.log'.format(name)
    final_log_file = os.path.join(logger_file_path, log_name)

    logger = logging.getLogger()  # 设定日志对象
    logger.setLevel(logging.INFO)  # 设定日志等级

    file_handler = logging.FileHandler(final_log_file)  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)  # 设置文件输出格式
    console_handler.setFormatter(formatter)  # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

mode = opt.mode
config_pth = opt.config_pth
ckpt_pth = opt.ckpt_pth

# save_path = f'/mnt/gestalt/home/lonian/mamba/test_results/transformer_1_cross_smalldata_bf16_10000steps'
with open(config_pth) as f:
    config = json.load(f)
model = Text_Mmamba_pl.load_from_checkpoint(ckpt_pth, config)

# /mnt/gestalt/home/lonian/mamba/model/ckpts/transformer_1_cross/lightning_logs/version_0/checkpoints/epoch=79-step=34000.ckpt

# with open('/mnt/gestalt/home/lonian/datasets/MusicBench/MusicBench_test_A_simba.json') as f:
#     meta_A = json.load(f)
# import datasets
# datasets_200 = datasets.load_from_disk('/mnt/gestalt/home/lonian/datasets/MusicCaps/eval_sub_200')
# loader = DataLoader(dataset=datasets_200, batch_size = 10)

# model = Text_Mmamba_pl.load_from_checkpoint("/mnt/gestalt/home/lonian/mamba/model/ckpts/dac_1_transformer/lightning_logs/version_3/checkpoints/epoch=61-step=25500.ckpt", config)

model.eval()
model.freeze()
folder_name = opt.folder_name
save_path = f'/mnt/gestalt/home/lonian/mamba/exp_results/{folder_name}/{mode}/dac_token'
os.makedirs (save_path, exist_ok=True)

logger = create_logger(f'/mnt/gestalt/home/lonian/mamba/exp_results/{folder_name}/{mode}', name=mode)

logger.info(f'Is incontext: {config['model']['is_incontext']}')
logger.info(f'Attention layers: {config['model']['self_atten_layers']}')
logger.info(f'Is pure mamba: {config['model']['is_pure_mamba']}')

L = 2588//3
with torch.no_grad():
    device = 'cuda'
    # for i in tqdm(loader):
    #     if os.path.isfile(os.path.join(save_path, '{}.npy'.format(i['ytid'][0]))):
    #         continue
    description = [ 'A vibrant and catchy melody drives this upbeat pop track, featuring smooth vocals layered over a backdrop of rhythmic claps, bright synths, and a steady drumbeat. The song exudes positivity and is perfect for a sunny day or a lively party.', 
                    'This mellow jazz piece blends a walking bassline with intricate piano improvisations and the soft hum of a saxophone solo. Its laid-back tempo and smooth swing rhythm create a cozy, sophisticated atmosphere ideal for a relaxed evening.', 
                    'A high-energy EDM track with thumping bass drops, euphoric synth leads, and an infectious rhythm. The buildup and release of tension keep the crowd on their feet, making it the ultimate choice for a late-night dance party.', 
                    'An explosive rock anthem powered by crunchy electric guitar riffs, a pulsating bassline, and dynamic drum fills. The raw, emotive vocals bring energy and intensity, making it an ideal soundtrack for a road trip or an adrenaline-pumping workout.', 
                    'A delicate and expressive piano solo, weaving a gentle melody with flowing arpeggios. The piece captures a contemplative mood, evoking imagery of a serene landscape at sunrise, perfect for moments of quiet reflection.',
                    'An experimental electronic piece blending glitchy beats, atmospheric textures, and robotic vocal effects. The track evolves through intricate soundscapes and pulsating basslines, creating a futuristic and immersive sound journey.']
    
    # description = [ 'A rhythmic electronic track with steady beats, shimmering synths, and a deep, pulsing bassline, creating an immersive, futuristic vibe.',
    #                 'A dreamy electronic composition filled with soft pads, airy melodies, and a gentle, flowing rhythm that feels ethereal and calming.',
    #                 'A high-energy electronic song featuring fast beats, glitchy effects, and bright synth leads that build an exciting, dynamic mood.',
    #                 'A minimalist electronic piece with repetitive beats, subtle textures, and a hypnotic, looping melody that feels introspective and atmospheric.',
    #                 'A vibrant electronic tune with layered arpeggios, sparkling synths, and a bouncy bass that creates an uplifting and playful sound.']
    
    # print(len(i['ytid']))
    prompt_seq = model(description=description, length=L, g_scale=3)
    print(prompt_seq.shape)

    for b in range(len(description)//2):
        gen_id = len(glob(f'{save_path}/*.npy')) + 1
        np.save(os.path.join(save_path, '{}.npy'.format(f'{gen_id}_{description[b].replace(' ', '_')}')), prompt_seq[b, :, :L])